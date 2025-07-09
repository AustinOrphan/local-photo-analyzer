"""Main CLI interface for photo analyzer."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from ..core.config import get_config
from ..core.logger import setup_logging, get_logger
from .advanced import advanced_cli
from ..database.engine import get_database_engine
from ..database.init import init_database, reset_database, check_database_health
from ..database.migrations import get_migration_manager
from ..pipeline.analyzer import PhotoAnalyzer
from ..pipeline.processor import PhotoProcessor
from ..pipeline.organizer import PhotoOrganizer

console = Console()
logger = get_logger(__name__)


async def analyze_standalone(config, image_files: List[Path], batch_size: int) -> List[Dict[str, Any]]:
    """Analyze photos without database import."""
    from ..analyzer.llm_client import OllamaClient
    from ..utils.image import ImageProcessor
    from ..utils.exif import ExifExtractor
    from ..pipeline.processor import PhotoProcessor
    import asyncio
    
    llm_client = OllamaClient(config)
    image_processor = ImageProcessor()
    exif_extractor = ExifExtractor()
    processor = PhotoProcessor(config)
    
    results = []
    total_files = len(image_files)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Analyzing photos (standalone)...", total=total_files)
        
        # Process in batches
        for i in range(0, total_files, batch_size):
            batch = image_files[i:i + batch_size]
            
            # Analyze batch concurrently
            batch_tasks = [
                analyze_single_standalone(
                    file_path, llm_client, image_processor, exif_extractor, processor
                )
                for file_path in batch
            ]
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to analyze {batch[j]}: {result}")
                        results.append({
                            'file_path': str(batch[j]),
                            'filename': batch[j].name,
                            'error': str(result),
                            'success': False
                        })
                    else:
                        result['file_path'] = str(batch[j])
                        result['filename'] = batch[j].name
                        result['success'] = True
                        results.append(result)
                
                progress.update(task, completed=len(results))
                
            except Exception as e:
                logger.error(f"Batch analysis failed: {e}")
                raise
    
    success_count = sum(1 for r in results if r.get('success', False))
    console.print(f"[green]Standalone analysis complete: {success_count}/{total_files} successful[/green]")
    
    return results


async def analyze_single_standalone(
    file_path: Path, 
    llm_client, 
    image_processor, 
    exif_extractor, 
    processor
) -> Dict[str, Any]:
    """Analyze a single photo in standalone mode."""
    try:
        # Validate image
        validation = image_processor.validate_image(file_path)
        if not validation['is_valid']:
            raise ValueError(f"Invalid image: {validation['errors']}")
        
        # Extract image metadata
        image_info = image_processor.get_image_info(file_path)
        exif_data = exif_extractor.extract_exif(file_path)
        
        # Perform LLM analysis
        if not await llm_client.check_connection():
            raise ConnectionError("LLM service is not available")
        
        llm_results = await llm_client.analyze_image(file_path)
        
        # Generate suggested filename
        suggested_filename = processor.generate_smart_filename(
            description=llm_results.get('description', ''),
            tags=llm_results.get('tags', []),
            date_taken=exif_data.get('date_taken')
        )
        
        # Extract camera and location info
        camera_info = extract_camera_info(exif_data)
        location_info = extract_location_info(exif_data)
        
        return {
            'description': llm_results.get('description', ''),
            'tags': llm_results.get('tags', []),
            'suggested_filename': suggested_filename,
            'confidence': llm_results.get('confidence', 0.0),
            'date_taken': exif_data.get('date_taken'),
            'camera_info': camera_info,
            'location': location_info,
            'image_properties': {
                'width': image_info['width'],
                'height': image_info['height'],
                'format': image_info['format'],
                'size_bytes': image_info['file_size'],
                'orientation': image_info['orientation'],
            },
            'processing_time': llm_results.get('processing_time', 0.0),
            'model_used': llm_results.get('model_used', ''),
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze photo {file_path}: {e}")
        return {
            'description': 'Analysis failed',
            'tags': [],
            'suggested_filename': file_path.name,
            'confidence': 0.0,
            'error': str(e)
        }


def extract_camera_info(exif_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract camera information from EXIF data."""
    if not exif_data:
        return None
    
    camera_info = {}
    
    if 'camera_make' in exif_data:
        camera_info['make'] = exif_data['camera_make']
    
    if 'camera_model' in exif_data:
        camera_info['model'] = exif_data['camera_model']
    
    if 'lens_model' in exif_data:
        camera_info['lens'] = exif_data['lens_model']
    
    # Settings
    settings = {}
    for key in ['iso', 'aperture', 'shutter_speed', 'focal_length']:
        if key in exif_data:
            settings[key] = exif_data[key]
    
    if settings:
        camera_info['settings'] = settings
    
    return camera_info if camera_info else None


def extract_location_info(exif_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract location information from EXIF data."""
    gps_data = exif_data.get('GPS', {})
    
    if 'latitude' in gps_data and 'longitude' in gps_data:
        location = {
            'latitude': gps_data['latitude'],
            'longitude': gps_data['longitude']
        }
        
        if 'altitude' in gps_data:
            location['altitude'] = gps_data['altitude']
        
        return location
    
    return None


def export_results(results: List[Dict[str, Any]], export_path: str, export_format: Optional[str]):
    """Export analysis results to file."""
    from pathlib import Path
    import json
    import csv
    from datetime import datetime
    
    export_file = Path(export_path)
    
    # Auto-detect format from extension if not specified
    if not export_format:
        ext = export_file.suffix.lower()
        if ext == '.csv':
            export_format = 'csv'
        elif ext == '.json':
            export_format = 'json'
        elif ext in ['.md', '.markdown']:
            export_format = 'markdown'
        else:
            export_format = 'json'  # Default
    
    try:
        if export_format == 'csv':
            export_csv(results, export_file)
        elif export_format == 'json':
            export_json(results, export_file)
        elif export_format == 'markdown':
            export_markdown(results, export_file)
        
        console.print(f"[green]Results exported to: {export_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]Failed to export results: {e}[/red]")


def export_csv(results: List[Dict[str, Any]], file_path: Path):
    """Export results to CSV format."""
    import csv
    
    with file_path.open('w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'filename', 'file_path', 'success', 'description', 'tags', 
            'suggested_filename', 'confidence', 'date_taken', 'width', 'height', 
            'format', 'size_bytes', 'camera_make', 'camera_model', 'iso', 
            'aperture', 'shutter_speed', 'focal_length', 'latitude', 'longitude',
            'processing_time', 'model_used', 'error'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {
                'filename': result.get('filename', ''),
                'file_path': result.get('file_path', ''),
                'success': result.get('success', False),
                'description': result.get('description', ''),
                'tags': ', '.join(result.get('tags', [])),
                'suggested_filename': result.get('suggested_filename', ''),
                'confidence': result.get('confidence', 0.0),
                'date_taken': result.get('date_taken', ''),
                'processing_time': result.get('processing_time', 0.0),
                'model_used': result.get('model_used', ''),
                'error': result.get('error', '')
            }
            
            # Image properties
            img_props = result.get('image_properties', {})
            row.update({
                'width': img_props.get('width', ''),
                'height': img_props.get('height', ''),
                'format': img_props.get('format', ''),
                'size_bytes': img_props.get('size_bytes', '')
            })
            
            # Camera info
            camera_info = result.get('camera_info', {})
            if camera_info:
                row.update({
                    'camera_make': camera_info.get('make', ''),
                    'camera_model': camera_info.get('model', '')
                })
                
                settings = camera_info.get('settings', {})
                row.update({
                    'iso': settings.get('iso', ''),
                    'aperture': settings.get('aperture', ''),
                    'shutter_speed': settings.get('shutter_speed', ''),
                    'focal_length': settings.get('focal_length', '')
                })
            
            # Location info
            location = result.get('location', {})
            if location:
                row.update({
                    'latitude': location.get('latitude', ''),
                    'longitude': location.get('longitude', '')
                })
            
            writer.writerow(row)


def export_json(results: List[Dict[str, Any]], file_path: Path):
    """Export results to JSON format."""
    import json
    from datetime import datetime
    
    export_data = {
        'export_timestamp': datetime.now().isoformat(),
        'total_photos': len(results),
        'successful_analyses': sum(1 for r in results if r.get('success', False)),
        'results': results
    }
    
    with file_path.open('w', encoding='utf-8') as jsonfile:
        json.dump(export_data, jsonfile, indent=2, default=str)


def export_markdown(results: List[Dict[str, Any]], file_path: Path):
    """Export results to Markdown format."""
    from datetime import datetime
    
    with file_path.open('w', encoding='utf-8') as mdfile:
        mdfile.write("# Photo Analysis Report\n\n")
        mdfile.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        mdfile.write(f"**Total Photos:** {len(results)}\n\n")
        mdfile.write(f"**Successful Analyses:** {sum(1 for r in results if r.get('success', False))}\n\n")
        
        mdfile.write("## Analysis Results\n\n")
        
        for i, result in enumerate(results, 1):
            mdfile.write(f"### {i}. {result.get('filename', 'Unknown')}\n\n")
            
            if not result.get('success', False):
                mdfile.write(f"**Status:** ❌ Failed\n\n")
                mdfile.write(f"**Error:** {result.get('error', 'Unknown error')}\n\n")
                continue
            
            mdfile.write(f"**Status:** ✅ Success\n\n")
            mdfile.write(f"**File Path:** `{result.get('file_path', '')}`\n\n")
            mdfile.write(f"**Description:** {result.get('description', 'No description')}\n\n")
            
            tags = result.get('tags', [])
            if tags:
                mdfile.write(f"**Tags:** {', '.join(f'`{tag}`' for tag in tags)}\n\n")
            
            mdfile.write(f"**Suggested Filename:** `{result.get('suggested_filename', '')}`\n\n")
            mdfile.write(f"**Confidence:** {result.get('confidence', 0.0):.2f}\n\n")
            
            # Image properties
            img_props = result.get('image_properties', {})
            if img_props:
                mdfile.write("**Image Properties:**\n")
                mdfile.write(f"- Dimensions: {img_props.get('width', 'N/A')} × {img_props.get('height', 'N/A')}\n")
                mdfile.write(f"- Format: {img_props.get('format', 'N/A')}\n")
                size_mb = img_props.get('size_bytes', 0) / (1024 * 1024)
                mdfile.write(f"- Size: {size_mb:.2f} MB\n\n")
            
            # Camera info
            camera_info = result.get('camera_info', {})
            if camera_info:
                mdfile.write("**Camera Information:**\n")
                if 'make' in camera_info:
                    mdfile.write(f"- Make: {camera_info['make']}\n")
                if 'model' in camera_info:
                    mdfile.write(f"- Model: {camera_info['model']}\n")
                
                settings = camera_info.get('settings', {})
                if settings:
                    mdfile.write("- Settings:\n")
                    for key, value in settings.items():
                        mdfile.write(f"  - {key.replace('_', ' ').title()}: {value}\n")
                mdfile.write("\n")
            
            # Location info
            location = result.get('location', {})
            if location:
                mdfile.write("**Location:**\n")
                mdfile.write(f"- Latitude: {location.get('latitude', 'N/A')}\n")
                mdfile.write(f"- Longitude: {location.get('longitude', 'N/A')}\n")
                if 'altitude' in location:
                    mdfile.write(f"- Altitude: {location['altitude']} m\n")
                mdfile.write("\n")
            
            mdfile.write("---\n\n")


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--config-file', type=click.Path(exists=True), help='Path to configuration file')
@click.pass_context
def main(ctx: click.Context, debug: bool, config_file: Optional[str]):
    """Photo Analyzer CLI - Secure local LLM-based photo analysis and organization.
    
    A privacy-first photo management system that uses local Large Language Models
    to analyze, tag, and organize your photos without sending data to external services.
    
    \b
    Key Features:
    • Local LLM analysis using Ollama/LLaVA
    • Intelligent photo tagging and description
    • Date-based organization with symbolic links
    • Smart filename generation
    • Manual tag management
    • Performance monitoring
    
    \b
    Quick Start:
    1. Initialize: photo-analyzer init
    2. Analyze photos: photo-analyzer analyze /path/to/photos
    3. Organize photos: photo-analyzer organize /path/to/photos /output/dir
    
    \b
    Examples:
    photo-analyzer analyze ~/Pictures --batch-size 3
    photo-analyzer organize ~/Pictures ~/Photos/organized --dry-run
    photo-analyzer tags add photo.jpg beautiful landscape --category nature
    photo-analyzer search "sunset ocean"
    """
    # Setup logging
    log_level = "DEBUG" if debug else "INFO"
    setup_logging(log_level)
    
    # Store context
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug
    ctx.obj['config_file'] = config_file
    
    # Load configuration
    try:
        config = get_config()
        ctx.obj['config'] = config
    except Exception as e:
        console.print(f"[red]Failed to load configuration: {e}[/red]")
        ctx.exit(1)


@main.command()
@click.option('--database-type', type=click.Choice(['sqlite', 'postgresql']), default='sqlite', 
              help='Database type to use (sqlite recommended for local use)')
@click.option('--database-path', type=click.Path(), help='Database file path (for SQLite)')
@click.option('--reset', is_flag=True, help='Reset existing database (WARNING: destroys all data)')
@click.pass_context
def init(ctx: click.Context, database_type: str, database_path: Optional[str], reset: bool):
    """Initialize the photo analyzer database and configuration.
    
    Sets up the database schema, creates necessary tables, and prepares
    the system for photo analysis. This should be run once before using
    other commands.
    
    \b
    Database Options:
    • sqlite (default): Local file-based database, no setup required
    • postgresql: Network database, requires separate PostgreSQL installation
    
    \b
    Examples:
    photo-analyzer init                    # Initialize with default SQLite
    photo-analyzer init --reset            # Reset database (destroys all data)
    photo-analyzer init --database-type postgresql
    
    \b
    Safety:
    The --reset flag will permanently delete all analyzed photos, tags,
    and organization data. Use with caution.
    """
    config = ctx.obj['config']
    
    console.print("[blue]Initializing Photo Analyzer...[/blue]")
    
    async def init_db():
        try:
            if reset:
                console.print("[yellow]Resetting database...[/yellow]")
                success = await reset_database()
            else:
                console.print("[blue]Initializing database...[/blue]")
                success = await init_database()
            
            if not success:
                raise RuntimeError("Database initialization failed")
            
            # Check health
            health = await check_database_health()
            if health['status'] == 'healthy':
                console.print(f"[green]Database initialized successfully![/green]")
                console.print(f"Database type: {health['database_type']}")
                console.print(f"Tables created: {len(health['tables'])}")
                
                # Show table counts
                if health['record_counts']:
                    table = Table(title="Database Status")
                    table.add_column("Table", style="cyan")
                    table.add_column("Records", style="green")
                    
                    for table_name, count in health['record_counts'].items():
                        table.add_row(table_name, str(count))
                    
                    console.print(table)
            else:
                raise RuntimeError(f"Database unhealthy: {health.get('error', 'Unknown error')}")
            
        except Exception as e:
            console.print(f"[red]Database initialization failed: {e}[/red]")
            raise
    
    # Run async initialization
    asyncio.run(init_db())
    console.print("[green]Photo Analyzer initialization complete![/green]")


@main.command()
@click.argument('paths', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--batch-size', default=5, help='Number of photos to process concurrently (1-20)')
@click.option('--output-format', type=click.Choice(['json', 'table']), default='table',
              help='Output format for results')
@click.option('--no-db', is_flag=True, help='Analyze without importing to database (standalone mode)')
@click.option('--export', type=click.Path(), help='Export results to file (CSV, JSON, or Markdown)')
@click.option('--export-format', type=click.Choice(['csv', 'json', 'markdown']), 
              help='Export format (auto-detected from file extension if not specified)')
@click.pass_context
def analyze(ctx: click.Context, paths: tuple, batch_size: int, output_format: str, no_db: bool, 
            export: Optional[str], export_format: Optional[str]):
    """Analyze photos using local LLM for content description and tagging.
    
    Uses a local Large Language Model (via Ollama) to analyze images and extract:
    • Detailed content descriptions
    • Relevant tags for categorization  
    • Smart filename suggestions
    • Confidence scores
    
    \b
    Analysis Modes:
    • Database mode (default): Store results for organization and search
    • Standalone mode (--no-db): Analyze without database storage
    
    \b
    Export Options:
    • CSV: Spreadsheet-friendly format with columns for metadata
    • JSON: Machine-readable format for scripting
    • Markdown: Human-readable format with formatted tables
    
    \b
    Supported Formats:
    • JPEG/JPG, PNG, TIFF, BMP, GIF
    • Searches recursively in directories
    
    \b
    Examples:
    photo-analyzer analyze photo.jpg                    # Standard analysis
    photo-analyzer analyze ~/Pictures --no-db          # No database import
    photo-analyzer analyze *.jpg --export results.csv  # Export to CSV
    photo-analyzer analyze /photos --export report.md --export-format markdown
    photo-analyzer analyze dir/ --no-db --export analysis.json --batch-size 3
    
    \b
    Prerequisites:
    • Ollama service running (http://localhost:11434)
    • LLaVA model installed (ollama pull llava)
    • Database initialized (only for database mode)
    """
    config = ctx.obj['config']
    
    # Collect all image files
    image_files = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_file():
            image_files.append(path)
        elif path.is_dir():
            # Find image files in directory
            for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
                image_files.extend(path.rglob(f'*{ext}'))
                image_files.extend(path.rglob(f'*{ext.upper()}'))
    
    if not image_files:
        console.print("[yellow]No image files found in specified paths[/yellow]")
        return
    
    console.print(f"[blue]Found {len(image_files)} image files to analyze[/blue]")
    
    if no_db:
        console.print("[yellow]Running in standalone mode (no database import)[/yellow]")
    
    async def analyze_photos():
        if no_db:
            # Standalone analysis without database
            results = await analyze_standalone(config, image_files, batch_size)
        else:
            # Standard database analysis
            analyzer = PhotoAnalyzer(config)
            results = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task("Analyzing photos...", total=len(image_files))
                
                def update_progress(completed: int, total: int):
                    progress.update(task, completed=completed)
                
                try:
                    results = await analyzer.analyze_batch(
                        image_files, 
                        batch_size=batch_size,
                        progress_callback=update_progress
                    )
                except Exception as e:
                    console.print(f"[red]Analysis failed: {e}[/red]")
                    return
        
        # Export results if requested
        if export:
            export_results(results, export, export_format)
        
        # Display results
        if output_format == 'table':
            display_analysis_table(results)
        else:
            import json
            console.print(json.dumps(results, indent=2, default=str))
    
    # Run analysis
    asyncio.run(analyze_photos())


@main.command()
@click.option('--failed-only', is_flag=True, help='Only reanalyze photos with failed or poor analysis results')
@click.option('--confidence-threshold', default=0.5, help='Reanalyze photos with confidence below this threshold')
@click.option('--force', is_flag=True, help='Force reanalysis of all photos regardless of existing results')
@click.option('--batch-size', default=3, help='Number of photos to process concurrently (1-10)')
@click.option('--cleanup', is_flag=True, help='Clean up old/redundant analysis results before reanalyzing')
@click.option('--dry-run', is_flag=True, help='Preview what would be reanalyzed without actually doing it')
@click.option('--photo-id', type=str, help='Reanalyze specific photo by ID')
@click.option('--output-format', type=click.Choice(['json', 'table']), default='table', help='Output format for results')
@click.pass_context
def reanalyze(ctx: click.Context, failed_only: bool, confidence_threshold: float, force: bool, 
              batch_size: int, cleanup: bool, dry_run: bool, photo_id: Optional[str], output_format: str):
    """Reanalyze photos to fix failed or poor analysis results.
    
    This command helps fix photos that had parsing failures, low confidence scores,
    or other analysis issues. It can:
    
    • Identify and reanalyze failed analysis results
    • Clean up redundant or corrupted analysis data
    • Force fresh analysis with improved prompts
    • Fix photos with placeholder or parsing error content
    
    \b
    Examples:
    photo-analyzer reanalyze --failed-only          # Only photos with obvious failures
    photo-analyzer reanalyze --confidence-threshold 0.7  # Low confidence photos
    photo-analyzer reanalyze --force                # Reanalyze all photos
    photo-analyzer reanalyze --photo-id abc123      # Specific photo
    photo-analyzer reanalyze --cleanup --dry-run    # Preview cleanup operations
    """
    
    from ..database.session import get_async_db_session
    from ..models.photo import Photo
    from ..models.analysis import AnalysisResult
    from sqlalchemy import select, and_, or_, func
    from sqlalchemy.orm import selectinload
    
    config = ctx.obj['config']
    
    if batch_size > 10:
        console.print("[yellow]Warning: Reducing batch size to 10 to avoid overwhelming the LLM service[/yellow]")
        batch_size = 10
    
    async def reanalyze_photos():
        async with get_async_db_session() as session:
            # Find photos that need reanalysis
            photos_to_reanalyze = []
            
            if photo_id:
                # Specific photo
                stmt = select(Photo).options(selectinload(Photo.analysis_results)).where(Photo.id == photo_id)
                result = await session.execute(stmt)
                photo = result.scalar_one_or_none()
                if photo:
                    photos_to_reanalyze = [photo]
                else:
                    console.print(f"[red]Photo with ID {photo_id} not found[/red]")
                    return
            else:
                # Build query based on criteria
                stmt = select(Photo).options(selectinload(Photo.analysis_results))
                
                if not force:
                    # Add filters for problematic photos
                    conditions = []
                    
                    if failed_only:
                        # Photos with obvious failures
                        conditions.append(
                            or_(
                                Photo.ai_description.is_(None),
                                Photo.ai_description == '',
                                Photo.ai_description.like('%json%'),
                                Photo.ai_description.like('%```%'),
                                Photo.ai_description.like('%description%'),
                                Photo.ai_description.like('%tags%'),
                                Photo.ai_description.like('%organization%')
                            )
                        )
                    else:
                        # Photos below confidence threshold or with poor descriptions
                        conditions.append(
                            or_(
                                Photo.ai_description.is_(None),
                                Photo.ai_description == '',
                                Photo.ai_description.like('%json%'),
                                Photo.ai_description.like('%```%')
                            )
                        )
                    
                    if conditions:
                        stmt = stmt.where(or_(*conditions))
                
                result = await session.execute(stmt)
                photos_to_reanalyze = result.scalars().all()
            
            if not photos_to_reanalyze:
                console.print("[green]No photos found that need reanalysis[/green]")
                return
            
            console.print(f"[blue]Found {len(photos_to_reanalyze)} photos for reanalysis[/blue]")
            
            if dry_run:
                # Show what would be reanalyzed
                table = Table(title="Photos to Reanalyze (Dry Run)")
                table.add_column("Filename", style="cyan")
                table.add_column("Current Description", style="dim")
                table.add_column("Confidence", justify="right")
                table.add_column("Analysis Results", justify="right")
                table.add_column("Issues", style="red")
                
                for photo in photos_to_reanalyze:
                    issues = []
                    if not photo.ai_description:
                        issues.append("No description")
                    elif any(word in photo.ai_description.lower() for word in ['json', '```', 'description', 'tags', 'organization']):
                        issues.append("Parsing errors")
                    
                    # Check confidence from most recent analysis result
                    if photo.analysis_results:
                        latest_result = max(photo.analysis_results, key=lambda x: x.created_at)
                        if latest_result.overall_confidence and latest_result.overall_confidence < confidence_threshold:
                            issues.append(f"Low confidence ({latest_result.overall_confidence:.2f})")
                    
                    if len(photo.analysis_results) > 3:
                        issues.append(f"Too many results ({len(photo.analysis_results)})")
                    
                    # Get confidence from latest analysis result
                    confidence_str = "None"
                    if photo.analysis_results:
                        latest_result = max(photo.analysis_results, key=lambda x: x.created_at)
                        if latest_result.overall_confidence is not None:
                            confidence_str = f"{latest_result.overall_confidence:.2f}"
                    
                    table.add_row(
                        photo.filename,
                        (photo.ai_description[:50] + "...") if photo.ai_description else "None",
                        confidence_str,
                        str(len(photo.analysis_results)),
                        ", ".join(issues) or "Unknown"
                    )
                
                console.print(table)
                return
            
            # Cleanup old results if requested
            if cleanup:
                console.print("[yellow]Cleaning up old analysis results...[/yellow]")
                cleanup_count = 0
                
                for photo in photos_to_reanalyze:
                    if len(photo.analysis_results) > 1:
                        # Keep only the most recent result, delete others
                        sorted_results = sorted(photo.analysis_results, key=lambda x: x.created_at, reverse=True)
                        to_delete = sorted_results[1:]  # Keep first (most recent), delete rest
                        
                        for result in to_delete:
                            await session.delete(result)
                            cleanup_count += 1
                
                if cleanup_count > 0:
                    await session.commit()
                    console.print(f"[green]Cleaned up {cleanup_count} old analysis results[/green]")
            
            # Perform reanalysis
            analyzer = PhotoAnalyzer(config)
            results = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Reanalyzing photos...", total=len(photos_to_reanalyze))
                
                # Process in batches
                for i in range(0, len(photos_to_reanalyze), batch_size):
                    batch = photos_to_reanalyze[i:i + batch_size]
                    
                    # Analyze batch concurrently
                    batch_tasks = []
                    for photo in batch:
                        task_coro = analyzer.analyze_photo(photo.current_path, session)
                        batch_tasks.append(task_coro)
                    
                    try:
                        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                        
                        for j, result in enumerate(batch_results):
                            photo = batch[j]
                            if isinstance(result, Exception):
                                console.print(f"[red]Failed to reanalyze {photo.filename}: {result}[/red]")
                                results.append({
                                    'filename': photo.filename,
                                    'photo_id': photo.id,
                                    'success': False,
                                    'error': str(result)
                                })
                            else:
                                results.append({
                                    'filename': photo.filename,
                                    'photo_id': photo.id,
                                    'success': True,
                                    'description': result.get('description', '')[:100] + '...',
                                    'suggested_filename': result.get('suggested_filename', ''),
                                    'confidence': result.get('confidence', 0.0),
                                    'tags_count': len(result.get('tags', []))
                                })
                            
                            progress.advance(task)
                    
                    except Exception as e:
                        console.print(f"[red]Batch processing failed: {e}[/red]")
                        break
            
            # Commit final changes
            await session.commit()
            
            success_count = sum(1 for r in results if r.get('success', False))
            console.print(f"[green]Reanalysis complete: {success_count}/{len(photos_to_reanalyze)} successful[/green]")
            
            # Display results
            if output_format == 'table' and results:
                table = Table(title="Reanalysis Results")
                table.add_column("Filename", style="cyan")
                table.add_column("Status", justify="center")
                table.add_column("New Description", style="dim")
                table.add_column("Confidence", justify="right")
                table.add_column("Tags", justify="right")
                
                for result in results[:20]:  # Show first 20 results
                    status = "[green]✓[/green]" if result['success'] else "[red]✗[/red]"
                    table.add_row(
                        result['filename'],
                        status,
                        result.get('description', result.get('error', 'N/A')),
                        f"{result.get('confidence', 0.0):.2f}" if result['success'] else "N/A",
                        str(result.get('tags_count', 0)) if result['success'] else "N/A"
                    )
                
                console.print(table)
                
                if len(results) > 20:
                    console.print(f"[dim]... and {len(results) - 20} more results[/dim]")
            
            elif output_format == 'json':
                import json
                console.print(json.dumps(results, indent=2, default=str))
    
    # Run reanalysis
    asyncio.run(reanalyze_photos())


@main.command()
@click.argument('paths', nargs=-1, required=True, type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--date-format', default='YYYY/MM/DD', 
              type=click.Choice(['YYYY/MM/DD', 'YYYY/MM', 'YYYY', 'YYYY/QN']),
              help='Date directory structure format')
@click.option('--create-symlinks/--no-symlinks', default=True, 
              help='Create categorical symbolic links for easy browsing')
@click.option('--filename-strategy', type=click.Choice(['smart', 'preserve']), default='smart',
              help='Filename strategy: smart (AI-generated) or preserve (original)')
@click.option('--dry-run', is_flag=True, help='Preview changes without actually moving files')
@click.option('--batch-size', default=5, help='Number of photos to process concurrently (1-20)')
@click.pass_context
def organize(ctx: click.Context, paths: tuple, output_dir: str, date_format: str, 
            create_symlinks: bool, filename_strategy: str, dry_run: bool, batch_size: int):
    """Organize photos into a structured directory hierarchy with symbolic links.
    
    Creates a date-based directory structure and moves photos from input paths
    to the organized output directory. Also creates categorical symbolic links
    for browsing by tags, camera, year, etc.
    
    \b
    Directory Structure Created:
    output_dir/
    ├── 2024/03/15/           # Date-based primary organization
    │   ├── sunset_ocean.jpg  # Smart filenames or preserved originals
    │   └── family_portrait.jpg
    ├── by_tags/              # Symbolic links by AI/manual tags
    │   ├── sunset/
    │   ├── family/
    │   └── portrait/
    ├── by_camera/            # Symbolic links by camera model
    │   ├── Canon_EOS_5D/
    │   └── iPhone_12/
    └── by_year/              # Symbolic links by year
        ├── 2023/
        └── 2024/
    
    \b
    Date Format Options:
    • YYYY/MM/DD: 2024/03/15 (default, most organized)
    • YYYY/MM: 2024/03 (by month)
    • YYYY: 2024 (by year only)
    • YYYY/QN: 2024/Q1 (by quarter)
    
    \b
    Filename Strategies:
    • smart: Use AI-generated descriptive names (sunset_ocean.jpg)
    • preserve: Keep original filenames (DSC_1234.jpg)
    
    \b
    Examples:
    photo-analyzer organize ~/Pictures ~/Photos/organized --dry-run
    photo-analyzer organize *.jpg /organized --filename-strategy preserve
    photo-analyzer organize /source /dest --date-format YYYY/MM --no-symlinks
    photo-analyzer organize /photos /organized --batch-size 3
    
    \b
    Prerequisites:
    • Photos must be analyzed first (photo-analyzer analyze)
    • Output directory will be created if it doesn't exist
    • Sufficient disk space for moving/copying files
    
    \b
    Safety:
    • Use --dry-run to preview changes before committing
    • Original files are moved, not copied (use with caution)
    • Symbolic links allow browsing without file duplication
    """
    config = ctx.obj['config']
    output_path = Path(output_dir)
    
    # Collect photos to organize
    photo_paths = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_file():
            photo_paths.append(path)
        elif path.is_dir():
            for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
                photo_paths.extend(path.rglob(f'*{ext}'))
                photo_paths.extend(path.rglob(f'*{ext.upper()}'))
    
    if not photo_paths:
        console.print("[yellow]No photos found to organize[/yellow]")
        return
    
    organization_rules = {
        'date_format': date_format,
        'filename_strategy': filename_strategy,
        'create_symlinks': create_symlinks,
        'symlink_categories': ['tags', 'camera', 'year', 'month'] if create_symlinks else [],
    }
    
    console.print(f"[blue]Organizing {len(photo_paths)} photos...[/blue]")
    if dry_run:
        console.print("[yellow]DRY RUN MODE - No changes will be made[/yellow]")
    
    async def organize_photos():
        organizer = PhotoOrganizer(config)
        processor = PhotoProcessor(config)
        
        # First, analyze photos if not already analyzed
        analyzer = PhotoAnalyzer(config)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # Analyze photos first
            analyze_task = progress.add_task("Analyzing photos...", total=len(photo_paths))
            
            def analyze_progress(completed: int, total: int):
                progress.update(analyze_task, completed=completed)
            
            analysis_results = await analyzer.analyze_batch(
                photo_paths,
                batch_size=batch_size,
                progress_callback=analyze_progress
            )
            
            # Get photo IDs for organization
            photo_ids = [r['photo_id'] for r in analysis_results if r.get('success')]
            
            if not photo_ids:
                console.print("[red]No photos were successfully analyzed[/red]")
                return
            
            # Organize photos
            organize_task = progress.add_task("Organizing photos...", total=len(photo_ids))
            
            def organize_progress(completed: int, total: int):
                progress.update(organize_task, completed=completed)
            
            organization_results = await organizer.organize_batch(
                photo_ids,
                output_path,
                organization_rules,
                max_concurrent=batch_size,
                progress_callback=organize_progress,
                dry_run=dry_run
            )
        
        # Display results
        display_organization_results(organization_results, dry_run)
    
    # Run organization
    asyncio.run(organize_photos())


@main.command()
@click.option('--hours', default=24, help='Time period for metrics (in hours)')
@click.option('--export', is_flag=True, help='Export detailed metrics to JSON file')
@click.pass_context
def performance(ctx: click.Context, hours: int, export: bool):
    """Display system performance metrics and optimization recommendations.
    
    Shows detailed performance statistics for photo analysis operations including
    memory usage, processing times, success rates, and system recommendations
    for optimal performance.
    
    \b
    Metrics Displayed:
    • Total operations performed
    • Success/failure rates  
    • Memory usage and deltas
    • CPU utilization
    • Average processing times per operation
    • Slow operations (>10 seconds)
    
    \b
    System Recommendations:
    • Memory optimization suggestions
    • Optimal batch size recommendations
    • CPU utilization improvements
    
    \b
    Examples:
    photo-analyzer performance                # Last 24 hours
    photo-analyzer performance --hours 72     # Last 3 days
    photo-analyzer performance --export       # Export to JSON file
    
    \b
    Performance Tips:
    • Lower batch sizes for systems with <4GB RAM
    • Higher batch sizes for multi-core systems
    • Monitor slow operations for optimization opportunities
    """
    config = ctx.obj['config']
    
    async def show_performance():
        from ..utils.performance import get_performance_monitor, PerformanceOptimizer
        
        monitor = get_performance_monitor()
        optimizer = PerformanceOptimizer()
        
        # Get metrics summary
        summary = monitor.get_metrics_summary(hours)
        system_info = optimizer.get_system_recommendations()
        
        console.print(f"\n[bold]Performance Summary (Last {hours} hours)[/bold]")
        
        # Overview table
        overview_table = Table(title="System Overview")
        overview_table.add_column("Metric", style="cyan")
        overview_table.add_column("Value", style="white")
        
        overview_table.add_row("Total Operations", str(summary['total_operations']))
        overview_table.add_row("Success Rate", f"{summary['overall_success_rate']:.1f}%")
        overview_table.add_row("Uptime", f"{summary['uptime_seconds'] / 3600:.1f} hours")
        overview_table.add_row("Memory Usage", f"{summary['current_memory_mb']:.1f} MB")
        overview_table.add_row("Memory Delta", f"{summary['memory_delta_mb']:+.1f} MB")
        overview_table.add_row("CPU Usage", f"{summary['current_cpu_percent']:.1f}%")
        
        console.print(overview_table)
        
        # Operations breakdown
        if summary['operations']:
            ops_table = Table(title="Operations Breakdown")
            ops_table.add_column("Operation", style="cyan")
            ops_table.add_column("Count", style="white")
            ops_table.add_column("Avg Duration", style="green")
            ops_table.add_column("Success Rate", style="yellow")
            ops_table.add_column("Avg Memory", style="blue")
            
            for op_name, stats in summary['operations'].items():
                ops_table.add_row(
                    op_name,
                    str(stats['count']),
                    f"{stats['avg_duration']:.2f}s",
                    f"{stats['success_rate']:.1f}%",
                    f"{stats['avg_memory']:+.1f}MB"
                )
            
            console.print(ops_table)
        
        # System recommendations
        if system_info.get('recommendations'):
            console.print("\n[bold]System Recommendations[/bold]")
            for rec in system_info['recommendations']:
                severity_color = {"warning": "yellow", "info": "blue", "error": "red"}.get(rec['severity'], "white")
                console.print(f"[{severity_color}]• {rec['message']}[/{severity_color}]")
                console.print(f"  💡 {rec['suggestion']}")
        
        # Slow operations
        slow_ops = monitor.get_slow_operations(threshold_seconds=10.0)
        if slow_ops:
            console.print(f"\n[yellow]⚠ Found {len(slow_ops)} slow operations (>10s)[/yellow]")
            for op in slow_ops[-3:]:  # Show last 3
                console.print(f"  • {op.operation}: {op.duration:.2f}s")
        
        # Export metrics if requested
        if export:
            export_path = monitor.export_metrics()
            console.print(f"\n[green]✓ Exported metrics to: {export_path}[/green]")
    
    asyncio.run(show_performance())


@main.command()
@click.pass_context
def status(ctx: click.Context):
    """Display system status and health check information.
    
    Shows the current status of all system components including database
    connectivity, migration status, LLM service availability, and overall
    system health.
    
    \b
    Status Components:
    • Database connection and table count
    • Migration status (applied vs pending)
    • LLM service connectivity (Ollama)
    • System configuration validation
    
    \b
    Examples:
    photo-analyzer status    # Show full system status
    
    \b
    Troubleshooting:
    • Database: Check if init was run successfully
    • Migrations: Look for pending database updates
    • LLM Service: Ensure Ollama is running on configured port
    """
    config = ctx.obj['config']
    
    async def show_status():
        try:
            # Database status
            db_engine = get_database_engine()
            table_names = await db_engine.get_table_names()
            
            # Migration status
            migration_manager = get_migration_manager()
            migration_status = await migration_manager.status()
            
            # Organization stats (if photos exist)
            organizer = PhotoOrganizer(config)
            # We'd need a base directory for this - skip for now or use config default
            
            # Create status table
            table = Table(title="Photo Analyzer Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details", style="white")
            
            # Database
            table.add_row("Database", "✓ Connected", f"{len(table_names)} tables")
            
            # Migrations
            table.add_row(
                "Migrations", 
                "✓ Up to date" if migration_status['pending_count'] == 0 else "⚠ Pending",
                f"{migration_status['applied_count']}/{migration_status['total_migrations']} applied"
            )
            
            # LLM Connection
            try:
                from ..analyzer.llm_client import OllamaClient
                llm_client = OllamaClient(config)
                health = await llm_client.check_connection()
                llm_status = "✓ Connected" if health else "✗ Unavailable"
            except Exception:
                llm_status = "✗ Error"
            
            table.add_row("LLM Service", llm_status, config.llm.ollama_url)
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Failed to get status: {e}[/red]")
    
    asyncio.run(show_status())


@main.command()
@click.argument('query', required=True)
@click.option('--limit', default=10, help='Maximum number of results to return (1-100)')
@click.option('--output-format', type=click.Choice(['json', 'table']), default='table',
              help='Output format for search results')
@click.pass_context
def search(ctx: click.Context, query: str, limit: int, output_format: str):
    """Search photos by content description, tags, filename, or metadata.
    
    Performs a comprehensive search across all analyzed photos using the provided
    query string. Searches through AI-generated descriptions, user descriptions,
    filenames, and associated tags.
    
    \b
    Search Targets:
    • AI-generated content descriptions
    • User-provided descriptions  
    • Photo filenames
    • Associated tags (both AI and manual)
    
    \b
    Search Tips:
    • Use descriptive terms: "sunset ocean" rather than "photo1"
    • Try tag names: "landscape", "family", "vacation"
    • Search camera info: "Canon", "iPhone"
    • Search dates or locations if available in metadata
    
    \b
    Examples:
    photo-analyzer search "sunset ocean"           # Content description
    photo-analyzer search "family vacation"       # Multiple terms
    photo-analyzer search "portrait" --limit 5    # Limit results
    photo-analyzer search "landscape" --output-format json
    
    \b
    Prerequisites:
    • Photos must be analyzed first (photo-analyzer analyze)
    • Better results with more descriptive AI analysis
    """
    config = ctx.obj['config']
    
    async def search_photos():
        from sqlalchemy import select, or_
        from ..database.session import get_async_db_session
        from ..models.photo import Photo, Tag
        
        async with get_async_db_session() as session:
            # Build search query
            stmt = select(Photo).where(
                or_(
                    Photo.ai_description.contains(query),
                    Photo.user_description.contains(query),
                    Photo.filename.contains(query),
                    Photo.tags.any(Tag.name.contains(query))
                )
            ).limit(limit)
            
            result = await session.execute(stmt)
            photos = result.scalars().all()
            
            if not photos:
                console.print("[yellow]No photos found matching query[/yellow]")
                return
            
            if output_format == 'table':
                display_search_results(photos, query)
            else:
                import json
                photo_data = [
                    {
                        'id': p.id,
                        'filename': p.filename,
                        'description': p.description,
                        'current_path': p.current_path,
                        'date_taken': p.date_taken.isoformat() if p.date_taken else None,
                        'tags': [tag.name for tag in p.tags] if p.tags else []
                    }
                    for p in photos
                ]
                console.print(json.dumps(photo_data, indent=2))
    
    asyncio.run(search_photos())


@main.command()
@click.option('--host', default='127.0.0.1', help='Host to bind to')
@click.option('--port', default=8000, type=int, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.option('--check-health', is_flag=True, help='Check LLM service health before starting')
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, reload: bool, check_health: bool):
    """Start the photo analyzer web service.
    
    Provides a web interface for photo analysis and management.
    
    \\b
    Examples:
    photo-analyzer serve                    # Start on default host/port
    photo-analyzer serve --host 0.0.0.0    # Bind to all interfaces
    photo-analyzer serve --port 8080       # Use custom port
    photo-analyzer serve --reload           # Enable auto-reload for development
    
    \\b
    Web Interface Features:
    - Photo upload and management
    - AI-powered analysis
    - Search and filtering
    - Tag management
    - System statistics
    """
    import uvicorn
    
    if check_health:
        async def check_llm_health():
            try:
                from ..analyzer.llm_client import OllamaClient
                config = ctx.obj['config']
                llm_client = OllamaClient(config)
                health = await llm_client.check_connection()
                
                if health:
                    console.print("[green]✓ LLM service is healthy[/green]")
                else:
                    console.print("[red]✗ LLM service is not responding[/red]")
                    console.print("[yellow]Warning: Some features may be unavailable[/yellow]")
            except Exception as e:
                console.print(f"[red]✗ LLM health check failed: {e}[/red]")
                console.print("[yellow]Warning: Some features may be unavailable[/yellow]")
        
        asyncio.run(check_llm_health())
    
    console.print(f"[green]Starting Photo Analyzer web server...[/green]")
    console.print(f"[blue]Host: {host}[/blue]")
    console.print(f"[blue]Port: {port}[/blue]")
    console.print(f"[blue]Reload: {reload}[/blue]")
    console.print(f"[cyan]Web Interface: http://{host}:{port}[/cyan]")
    console.print(f"[cyan]API Documentation: http://{host}:{port}/docs[/cyan]")
    console.print(f"[cyan]Health Check: http://{host}:{port}/health[/cyan]")
    console.print("")
    console.print("[yellow]Press Ctrl+C to stop the server[/yellow]")
    
    try:
        uvicorn.run(
            "photo_analyzer.web.app:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        console.print("[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        raise click.ClickException(f"Failed to start web server: {e}")


def display_analysis_table(results: List[dict]):
    """Display analysis results in a table."""
    table = Table(title="Photo Analysis Results")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Tags", style="green")
    table.add_column("Confidence", style="yellow")
    table.add_column("Status", style="magenta")
    
    for result in results:
        if result.get('success'):
            file_path = Path(result.get('file_path', ''))
            description = result.get('description', '')[:50] + "..." if len(result.get('description', '')) > 50 else result.get('description', '')
            tags = ", ".join(result.get('tags', []))[:30] + "..." if len(", ".join(result.get('tags', []))) > 30 else ", ".join(result.get('tags', []))
            confidence = f"{result.get('confidence', 0):.2f}"
            status = "✓ Success"
        else:
            file_path = Path(result.get('file_path', ''))
            description = "Error"
            tags = ""
            confidence = "0.00"
            status = f"✗ {result.get('error', 'Unknown error')}"
        
        table.add_row(
            file_path.name,
            description,
            tags,
            confidence,
            status
        )
    
    console.print(table)


def display_organization_results(results: List[dict], dry_run: bool):
    """Display organization results."""
    title = "Organization Results (DRY RUN)" if dry_run else "Organization Results"
    table = Table(title=title)
    table.add_column("Photo ID", style="cyan")
    table.add_column("Target Path", style="white")
    table.add_column("Symlinks", style="green")
    table.add_column("Status", style="magenta")
    
    for result in results:
        if result.get('success'):
            photo_id = result.get('photo_id', '')[:8] + "..."
            target_path = str(Path(result.get('target_path', '')).name)
            symlink_count = len(result.get('symlinks', []))
            symlinks = f"{symlink_count} links"
            status = "✓ Success"
        else:
            photo_id = result.get('photo_id', '')[:8] + "..."
            target_path = "Error"
            symlinks = ""
            status = f"✗ {result.get('error', 'Unknown error')}"
        
        table.add_row(photo_id, target_path, symlinks, status)
    
    console.print(table)


def display_search_results(photos: List, query: str):
    """Display search results in a table."""
    table = Table(title=f"Search Results for '{query}'")
    table.add_column("Filename", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Tags", style="green")
    table.add_column("Date Taken", style="yellow")
    table.add_column("Path", style="magenta")
    
    for photo in photos:
        filename = photo.filename
        description = (photo.ai_description[:40] + "...") if photo.ai_description and len(photo.ai_description) > 40 else (photo.ai_description or "")
        tags = ", ".join([tag.name for tag in photo.tags]) if photo.tags else ""
        date_taken = photo.date_taken.strftime("%Y-%m-%d") if photo.date_taken else ""
        path = str(Path(photo.current_path).parent)
        
        table.add_row(filename, description, tags, date_taken, path)
    
    console.print(table)


@main.command()
@click.argument('paths', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--dry-run', is_flag=True, help='Preview filename changes without applying them')
@click.option('--batch-size', default=5, help='Number of files to process concurrently (1-20)')
@click.pass_context
def rename(ctx: click.Context, paths: tuple, dry_run: bool, batch_size: int):
    """Apply AI-suggested smart filenames to analyzed photos.
    
    Renames photos using AI-generated descriptive filenames based on content
    analysis. Only works on photos that have been analyzed and have filename
    suggestions in the database.
    
    \b
    Filename Generation:
    • Based on AI content analysis
    • Uses descriptive terms from image content
    • Includes relevant tags and context
    • Preserves original file extensions
    • Handles filename conflicts automatically
    
    \b
    Examples:
    photo-analyzer rename photo.jpg --dry-run      # Preview single file
    photo-analyzer rename ~/Pictures --dry-run     # Preview directory
    photo-analyzer rename *.jpg                    # Rename multiple files
    photo-analyzer rename /photos --batch-size 3   # Process in small batches
    
    \b
    Safety Features:
    • Use --dry-run to preview changes first
    • Updates database records to match new filenames
    • Handles duplicate names automatically
    • Preserves original extensions
    
    \b
    Prerequisites:
    • Photos must be analyzed first (photo-analyzer analyze)
    • Photos must be in the database with analysis results
    • AI analysis must have generated filename suggestions
    """
    config = ctx.obj['config']
    
    async def rename_photos():
        from ..database.session import get_async_db_session
        from ..models.photo import Photo
        from sqlalchemy import select
        
        # Find all analyzed photos
        photos_to_rename = []
        
        for path_str in paths:
            path = Path(path_str)
            if path.is_file():
                photos_to_rename.append(path)
            elif path.is_dir():
                for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
                    photos_to_rename.extend(path.rglob(f'*{ext}'))
                    photos_to_rename.extend(path.rglob(f'*{ext.upper()}'))
        
        if not photos_to_rename:
            console.print("[yellow]No photos found in specified paths[/yellow]")
            return
        
        renamed_count = 0
        
        async with get_async_db_session() as session:
            for photo_path in photos_to_rename:
                # Find photo in database
                stmt = select(Photo).where(Photo.current_path == str(photo_path))
                result = await session.execute(stmt)
                photo = result.scalar_one_or_none()
                
                if not photo:
                    console.print(f"[yellow]Photo not in database: {photo_path.name}[/yellow]")
                    continue
                
                # Get the latest analysis result
                from ..models.analysis import AnalysisResult
                analysis_stmt = select(AnalysisResult).where(
                    AnalysisResult.photo_id == photo.id
                ).order_by(AnalysisResult.created_at.desc()).limit(1)
                analysis_result_query = await session.execute(analysis_stmt)
                latest_analysis = analysis_result_query.scalar_one_or_none()
                
                if latest_analysis and latest_analysis.suggested_filename:
                    suggested_name = latest_analysis.suggested_filename
                    
                    # Add extension if missing
                    if not suggested_name.endswith(photo_path.suffix):
                        suggested_name += photo_path.suffix
                    
                    new_path = photo_path.parent / suggested_name
                    
                    if new_path != photo_path:
                        if dry_run:
                            console.print(f"[blue]Would rename:[/blue] {photo_path.name} → {suggested_name}")
                        else:
                            try:
                                # Rename the file
                                photo_path.rename(new_path)
                                
                                # Update database
                                photo.current_path = str(new_path)
                                photo.filename = suggested_name
                                photo.updated_at = datetime.utcnow()
                                
                                console.print(f"[green]Renamed:[/green] {photo_path.name} → {suggested_name}")
                                renamed_count += 1
                            except Exception as e:
                                console.print(f"[red]Failed to rename {photo_path.name}: {e}[/red]")
                    else:
                        console.print(f"[dim]No change needed: {photo_path.name}[/dim]")
                else:
                    console.print(f"[yellow]No AI filename suggestion for: {photo_path.name}[/yellow]")
            
            if not dry_run:
                await session.commit()
        
        if dry_run:
            console.print(f"\n[blue]Dry run complete. Found {len(photos_to_rename)} photos to process.[/blue]")
        else:
            console.print(f"\n[green]Renamed {renamed_count} photos successfully.[/green]")
    
    asyncio.run(rename_photos())


@main.command('rename-standalone')
@click.argument('analysis_file', type=click.Path(exists=True))
@click.option('--dry-run', is_flag=True, help='Preview filename changes without applying them')
@click.option('--filter', type=str, help='Filter photos by filename pattern (e.g., "*.jpg")')
@click.pass_context
def rename_standalone(ctx: click.Context, analysis_file: str, dry_run: bool, filter: Optional[str]):
    """Rename photos based on exported analysis results without database.
    
    Uses previously exported analysis results (JSON, CSV, or Markdown) to rename
    photos with AI-suggested filenames. Works independently of the database,
    making it useful for batch operations or external workflows.
    
    \b
    Supported Analysis Files:
    • JSON exports from --export option
    • CSV exports with suggested_filename column
    • Compatible with standalone analysis results
    
    \b
    Features:
    • Renames files based on suggested_filename field
    • Preserves original file extensions
    • Handles filename conflicts automatically
    • Optional filename filtering with glob patterns
    
    \b
    Examples:
    photo-analyzer rename-standalone analysis.json --dry-run  # Preview changes
    photo-analyzer rename-standalone results.csv             # Apply renames
    photo-analyzer rename-standalone report.json --filter "IMG_*.jpg"
    
    \b
    Safety Features:
    • Use --dry-run to preview changes first
    • Automatically handles duplicate filenames
    • Preserves original extensions
    • Validates file paths before renaming
    
    \b
    Prerequisites:
    • Analysis results file from previous export
    • Write permissions to photo directories
    • Photos must still exist at original paths
    """
    import json
    import csv
    from pathlib import Path
    import fnmatch
    
    analysis_path = Path(analysis_file)
    
    async def rename_from_analysis():
        # Load analysis results
        try:
            if analysis_path.suffix.lower() == '.json':
                results = load_json_analysis(analysis_path)
            elif analysis_path.suffix.lower() == '.csv':
                results = load_csv_analysis(analysis_path)
            else:
                console.print(f"[red]Unsupported file format: {analysis_path.suffix}[/red]")
                console.print("[dim]Supported formats: .json, .csv[/dim]")
                return
        except Exception as e:
            console.print(f"[red]Failed to load analysis file: {e}[/red]")
            return
        
        # Filter results if pattern provided
        if filter:
            original_count = len(results)
            results = [r for r in results if fnmatch.fnmatch(r.get('filename', ''), filter)]
            console.print(f"[blue]Filtered to {len(results)} photos (from {original_count})[/blue]")
        
        if not results:
            console.print("[yellow]No photos found in analysis results[/yellow]")
            return
        
        renamed_count = 0
        failed_count = 0
        
        for result in results:
            if not result.get('success', False):
                continue
            
            file_path = Path(result.get('file_path', ''))
            suggested_name = result.get('suggested_filename', '')
            
            if not file_path.exists():
                console.print(f"[yellow]File not found: {file_path}[/yellow]")
                failed_count += 1
                continue
            
            if not suggested_name or suggested_name == file_path.name:
                console.print(f"[dim]No rename needed: {file_path.name}[/dim]")
                continue
            
            # Ensure extension is preserved
            if not suggested_name.endswith(file_path.suffix):
                suggested_name += file_path.suffix
            
            new_path = file_path.parent / suggested_name
            
            # Handle conflicts
            if new_path.exists() and new_path != file_path:
                new_path = resolve_filename_conflict(new_path)
                suggested_name = new_path.name
            
            if dry_run:
                console.print(f"[blue]Would rename:[/blue] {file_path.name} → {suggested_name}")
            else:
                try:
                    file_path.rename(new_path)
                    console.print(f"[green]Renamed:[/green] {file_path.name} → {suggested_name}")
                    renamed_count += 1
                except Exception as e:
                    console.print(f"[red]Failed to rename {file_path.name}: {e}[/red]")
                    failed_count += 1
        
        if dry_run:
            console.print(f"\n[blue]Dry run complete. Found {len(results)} photos to process.[/blue]")
        else:
            console.print(f"\n[green]Renamed {renamed_count} photos successfully.[/green]")
            if failed_count > 0:
                console.print(f"[yellow]{failed_count} operations failed.[/yellow]")
    
    asyncio.run(rename_from_analysis())


def load_json_analysis(file_path: Path) -> List[Dict[str, Any]]:
    """Load analysis results from JSON file."""
    import json
    
    with file_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'results' in data:
        return data['results']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Invalid JSON format - expected list or object with 'results' key")


def load_csv_analysis(file_path: Path) -> List[Dict[str, Any]]:
    """Load analysis results from CSV file."""
    import csv
    
    results = []
    with file_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert string boolean to actual boolean
            success = row.get('success', '').lower() in ('true', '1', 'yes')
            
            result = {
                'filename': row.get('filename', ''),
                'file_path': row.get('file_path', ''),
                'success': success,
                'suggested_filename': row.get('suggested_filename', ''),
                'description': row.get('description', ''),
                'tags': row.get('tags', '').split(', ') if row.get('tags') else [],
                'confidence': float(row.get('confidence', 0)) if row.get('confidence') else 0.0
            }
            results.append(result)
    
    return results


def resolve_filename_conflict(target_path: Path) -> Path:
    """Resolve filename conflicts by appending number."""
    counter = 1
    stem = target_path.stem
    suffix = target_path.suffix
    parent = target_path.parent
    
    while target_path.exists():
        new_name = f"{stem}_{counter}{suffix}"
        target_path = parent / new_name
        counter += 1
    
    return target_path


@main.group()
def tags():
    """Manage photo tags and categorization.
    
    Comprehensive tag management system supporting both AI-generated and manual tags.
    Tags help categorize and organize photos for better searchability and organization.
    
    \b
    Tag Features:
    • AI-generated tags from image analysis
    • Manual tag addition and removal
    • Automatic categorization (nature, people, objects, activities, general)
    • Usage statistics and analytics
    • Tag filtering and search capabilities
    
    \b
    Tag Categories:
    • nature: landscape, mountain, tree, forest, ocean, sky, sunset, sunrise
    • people: person, people, family, child, adult, portrait
    • objects: car, building, house, food, animal, dog, cat
    • activities: sports, running, swimming, hiking, travel, vacation
    • general: everything else
    
    \b
    Common Workflows:
    1. Analyze photos to get AI tags: photo-analyzer analyze /photos
    2. Add manual tags: photo-analyzer tags add photo.jpg beautiful peaceful
    3. View photo tags: photo-analyzer tags list photo.jpg  
    4. Remove unwanted tags: photo-analyzer tags remove photo.jpg unwanted
    5. View statistics: photo-analyzer tags stats
    """
    pass


@tags.command('add')
@click.argument('photo_path', type=click.Path(exists=True))
@click.argument('tag_names', nargs=-1, required=True)
@click.option('--category', type=str, 
              help='Override automatic categorization (nature, people, objects, activities, general)')
@click.option('--confidence', type=float, default=1.0, help='Confidence score for manual tags (0.0-1.0)')
@click.pass_context
def add_tags(ctx: click.Context, photo_path: str, tag_names: tuple, category: Optional[str], confidence: float):
    """Add manual tags to a photo for better categorization and search.
    
    Adds one or more tags to a photo that has been analyzed and stored in the
    database. Tags are marked as manually added (not AI-generated) and can be
    used for organization, search, and symbolic link creation.
    
    \b
    Tag Processing:
    • Automatically converts to lowercase and trims whitespace
    • Creates new tags if they don't exist in database
    • Associates tags with the specified photo
    • Updates tag usage statistics
    • Prevents duplicate tag assignments
    
    \b
    Categories (auto-assigned unless overridden):
    • nature: landscape, sunset, sky, tree, mountain, forest, ocean
    • people: portrait, family, person, child, adult
    • objects: house, car, building, food, animal
    • activities: sports, travel, vacation, hiking
    • general: everything else
    
    \b
    Examples:
    photo-analyzer tags add photo.jpg beautiful        # Single tag
    photo-analyzer tags add photo.jpg peaceful serene  # Multiple tags
    photo-analyzer tags add photo.jpg portrait --category people
    photo-analyzer tags add landscape.jpg dramatic --confidence 0.9
    
    \b
    Prerequisites:
    • Photo must be analyzed first (photo-analyzer analyze)
    • Photo must exist in database
    
    \b
    Notes:
    • Manual tags are distinguished from AI-generated tags
    • Tags can be used in organization symbolic links
    • Existing tags are not duplicated, only new ones added
    """
    async def add_tags_to_photo():
        from ..database.session import get_async_db_session
        from ..models.photo import Photo, Tag
        from sqlalchemy import select
        import uuid
        
        photo_path_obj = Path(photo_path)
        
        async with get_async_db_session() as session:
            # Find photo in database
            stmt = select(Photo).where(Photo.current_path == str(photo_path_obj))
            result = await session.execute(stmt)
            photo = result.scalar_one_or_none()
            
            if not photo:
                console.print(f"[red]Photo not found in database: {photo_path_obj.name}[/red]")
                console.print("[dim]Tip: Run 'analyze' command first to add photo to database[/dim]")
                return
            
            added_tags = []
            existing_tags = []
            
            for tag_name in tag_names:
                tag_name = tag_name.strip().lower()
                
                if not tag_name:
                    continue
                
                # Check if tag already exists
                tag_stmt = select(Tag).where(Tag.name == tag_name)
                tag_result = await session.execute(tag_stmt)
                tag = tag_result.scalar_one_or_none()
                
                if not tag:
                    # Create new tag
                    tag_category = category or _categorize_tag(tag_name)
                    tag = Tag(
                        id=str(uuid.uuid4()),
                        name=tag_name,
                        category=tag_category,
                        is_auto_generated=False,
                        usage_count=0,
                        created_at=datetime.utcnow()
                    )
                    session.add(tag)
                    await session.flush()
                    console.print(f"[green]Created new tag:[/green] {tag_name} ({tag_category})")
                
                # Check if photo already has this tag
                if tag not in photo.tags:
                    photo.tags.append(tag)
                    tag.usage_count += 1
                    added_tags.append(tag_name)
                else:
                    existing_tags.append(tag_name)
            
            # Update photo metadata
            if added_tags:
                photo.tag_count = len(photo.tags)
                photo.updated_at = datetime.utcnow()
            
            await session.commit()
            
            # Display results
            if added_tags:
                console.print(f"[green]Added {len(added_tags)} tags to {photo_path_obj.name}:[/green]")
                for tag_name in added_tags:
                    console.print(f"  + {tag_name}")
            
            if existing_tags:
                console.print(f"[yellow]Tags already exist:[/yellow]")
                for tag_name in existing_tags:
                    console.print(f"  ~ {tag_name}")
            
            if not added_tags and not existing_tags:
                console.print("[yellow]No valid tags provided[/yellow]")
    
    asyncio.run(add_tags_to_photo())


@tags.command('remove')
@click.argument('photo_path', type=click.Path(exists=True))
@click.argument('tag_names', nargs=-1, required=True)
@click.pass_context
def remove_tags(ctx: click.Context, photo_path: str, tag_names: tuple):
    """Remove specific tags from a photo.
    
    Removes one or more tags from a photo's tag associations. This only removes
    the association between the photo and the tag; the tag itself remains in
    the database for use with other photos.
    
    \b
    Tag Removal Process:
    • Removes tag associations from the specified photo
    • Updates tag usage statistics (decrements usage count)
    • Preserves tag definitions for other photos
    • Updates photo metadata and timestamps
    
    \b
    Examples:
    photo-analyzer tags remove photo.jpg unwanted      # Single tag
    photo-analyzer tags remove photo.jpg old boring    # Multiple tags
    photo-analyzer tags remove landscape.jpg person    # Remove incorrect AI tag
    
    \b
    Safety Features:
    • Only removes tags that are actually associated with the photo
    • Reports which tags were successfully removed
    • Reports which tags were not found on the photo
    • Maintains data integrity with usage counts
    
    \b
    Prerequisites:
    • Photo must exist in database
    • Tags must be currently associated with the photo
    
    \b
    Notes:
    • Can remove both AI-generated and manually added tags
    • Useful for correcting incorrect AI tagging
    • Tags remain available for other photos
    """
    async def remove_tags_from_photo():
        from ..database.session import get_async_db_session
        from ..models.photo import Photo, Tag
        from sqlalchemy import select
        
        photo_path_obj = Path(photo_path)
        
        async with get_async_db_session() as session:
            # Find photo in database
            stmt = select(Photo).where(Photo.current_path == str(photo_path_obj))
            result = await session.execute(stmt)
            photo = result.scalar_one_or_none()
            
            if not photo:
                console.print(f"[red]Photo not found in database: {photo_path_obj.name}[/red]")
                return
            
            removed_tags = []
            not_found_tags = []
            
            for tag_name in tag_names:
                tag_name = tag_name.strip().lower()
                
                # Find tag in photo's tags
                tag_to_remove = None
                for tag in photo.tags:
                    if tag.name == tag_name:
                        tag_to_remove = tag
                        break
                
                if tag_to_remove:
                    photo.tags.remove(tag_to_remove)
                    tag_to_remove.usage_count = max(0, tag_to_remove.usage_count - 1)
                    removed_tags.append(tag_name)
                else:
                    not_found_tags.append(tag_name)
            
            # Update photo metadata
            if removed_tags:
                photo.tag_count = len(photo.tags)
                photo.updated_at = datetime.utcnow()
            
            await session.commit()
            
            # Display results
            if removed_tags:
                console.print(f"[green]Removed {len(removed_tags)} tags from {photo_path_obj.name}:[/green]")
                for tag_name in removed_tags:
                    console.print(f"  - {tag_name}")
            
            if not_found_tags:
                console.print(f"[yellow]Tags not found on photo:[/yellow]")
                for tag_name in not_found_tags:
                    console.print(f"  ? {tag_name}")
    
    asyncio.run(remove_tags_from_photo())


@tags.command('list')
@click.argument('photo_path', type=click.Path(exists=True), required=False)
@click.option('--all', 'list_all', is_flag=True, help='List all tags in database instead of photo-specific tags')
@click.option('--category', type=str, help='Filter tags by category (nature, people, objects, activities, general)')
@click.option('--min-usage', type=int, default=0, help='Show only tags used by at least N photos')
@click.pass_context
def list_tags(ctx: click.Context, photo_path: Optional[str], list_all: bool, category: Optional[str], min_usage: int):
    """List tags for a specific photo or browse all tags in the database.
    
    Displays tag information in a formatted table showing tag names, categories,
    usage statistics, and source (AI-generated vs manually added).
    
    \b
    Display Modes:
    • Photo-specific: Show tags associated with a single photo
    • All tags: Show complete tag database with filtering options
    
    \b
    Tag Information Shown:
    • Tag name (normalized lowercase)
    • Category (nature, people, objects, activities, general)
    • Usage count (number of photos with this tag)
    • Source (AI-generated or Manual)
    
    \b
    Filtering Options:
    • Category: Show only tags in specific category
    • Min-usage: Show only popular tags (used by multiple photos)
    • Sorted by usage count (most used first)
    
    \b
    Examples:
    photo-analyzer tags list photo.jpg              # Tags for specific photo
    photo-analyzer tags list --all                  # All tags in database
    photo-analyzer tags list --all --category nature   # Nature tags only
    photo-analyzer tags list --all --min-usage 2       # Popular tags only
    
    \b
    Use Cases:
    • Review tags assigned to a photo
    • Browse available tags for manual assignment
    • Find popular tags for organization
    • Identify unused or rarely used tags
    """
    async def list_photo_tags():
        from ..database.session import get_async_db_session
        from ..models.photo import Photo, Tag
        from sqlalchemy import select
        
        async with get_async_db_session() as session:
            if photo_path:
                # List tags for specific photo
                photo_path_obj = Path(photo_path)
                stmt = select(Photo).where(Photo.current_path == str(photo_path_obj))
                result = await session.execute(stmt)
                photo = result.scalar_one_or_none()
                
                if not photo:
                    console.print(f"[red]Photo not found in database: {photo_path_obj.name}[/red]")
                    return
                
                if not photo.tags:
                    console.print(f"[yellow]No tags found for {photo_path_obj.name}[/yellow]")
                    return
                
                # Display photo tags
                table = Table(title=f"Tags for {photo_path_obj.name}")
                table.add_column("Tag", style="cyan")
                table.add_column("Category", style="green")
                table.add_column("Source", style="yellow")
                
                for tag in photo.tags:
                    source = "AI" if tag.is_auto_generated else "Manual"
                    table.add_row(tag.name, tag.category or "general", source)
                
                console.print(table)
                console.print(f"\n[blue]Total: {len(photo.tags)} tags[/blue]")
                
            elif list_all:
                # List all tags in database
                stmt = select(Tag)
                
                # Apply filters
                if category:
                    stmt = stmt.where(Tag.category == category)
                if min_usage > 0:
                    stmt = stmt.where(Tag.usage_count >= min_usage)
                
                stmt = stmt.order_by(Tag.usage_count.desc(), Tag.name)
                
                result = await session.execute(stmt)
                tags = result.scalars().all()
                
                if not tags:
                    console.print("[yellow]No tags found[/yellow]")
                    return
                
                # Display all tags
                table = Table(title="All Tags")
                table.add_column("Tag", style="cyan")
                table.add_column("Category", style="green")
                table.add_column("Usage", style="blue")
                table.add_column("Source", style="yellow")
                
                for tag in tags:
                    source = "AI" if tag.is_auto_generated else "Manual"
                    table.add_row(
                        tag.name, 
                        tag.category or "general", 
                        str(tag.usage_count),
                        source
                    )
                
                console.print(table)
                console.print(f"\n[blue]Total: {len(tags)} tags[/blue]")
                
            else:
                console.print("[red]Please specify a photo path or use --all flag[/red]")
    
    asyncio.run(list_photo_tags())


@tags.command('stats')
@click.pass_context
def tag_stats(ctx: click.Context):
    """Display comprehensive tag usage statistics and analytics.
    
    Provides detailed insights into tag usage patterns, distribution across
    categories, and the most popular tags in your photo collection.
    
    \b
    Statistics Included:
    • Total number of tags in database
    • Breakdown of AI-generated vs manually added tags
    • Top 10 most frequently used tags
    • Tag distribution by category
    • Usage patterns and trends
    
    \b
    Category Analytics:
    • Number of tags in each category
    • Most popular categories
    • Category distribution visualization
    
    \b
    Use Cases:
    • Understand tagging patterns in your collection
    • Identify over-used or under-used tags
    • Review AI tagging effectiveness
    • Plan manual tagging strategies
    • Optimize organization workflows
    
    \b
    Examples:
    photo-analyzer tags stats    # Complete tag analytics
    
    \b
    Insights Provided:
    • Which tags are most useful for organization
    • Balance between AI and manual tagging
    • Category trends in your photo collection
    • Opportunities for better tagging
    """
    async def show_tag_statistics():
        from ..database.session import get_async_db_session
        from ..models.photo import Tag
        from sqlalchemy import select, func
        
        async with get_async_db_session() as session:
            # Overall stats
            total_tags_stmt = select(func.count(Tag.id))
            total_tags = await session.scalar(total_tags_stmt) or 0
            
            ai_tags_stmt = select(func.count(Tag.id)).where(Tag.is_auto_generated == True)
            ai_tags = await session.scalar(ai_tags_stmt) or 0
            
            manual_tags = total_tags - ai_tags
            
            # Top used tags
            top_tags_stmt = select(Tag).order_by(Tag.usage_count.desc()).limit(10)
            top_tags_result = await session.execute(top_tags_stmt)
            top_tags = top_tags_result.scalars().all()
            
            # Category breakdown
            category_stmt = select(Tag.category, func.count(Tag.id)).group_by(Tag.category)
            category_result = await session.execute(category_stmt)
            categories = category_result.all()
            
            # Display overall stats
            stats_table = Table(title="Tag Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Count", style="green")
            
            stats_table.add_row("Total Tags", str(total_tags))
            stats_table.add_row("AI Generated", str(ai_tags))
            stats_table.add_row("Manual", str(manual_tags))
            
            console.print(stats_table)
            
            # Display top tags
            if top_tags:
                console.print("\n[bold]Top Used Tags[/bold]")
                top_table = Table()
                top_table.add_column("Tag", style="cyan")
                top_table.add_column("Usage", style="blue")
                top_table.add_column("Category", style="green")
                
                for tag in top_tags:
                    top_table.add_row(tag.name, str(tag.usage_count), tag.category or "general")
                
                console.print(top_table)
            
            # Display category breakdown
            if categories:
                console.print("\n[bold]Tags by Category[/bold]")
                cat_table = Table()
                cat_table.add_column("Category", style="cyan")
                cat_table.add_column("Count", style="blue")
                
                for category, count in categories:
                    cat_table.add_row(category or "general", str(count))
                
                console.print(cat_table)
    
    asyncio.run(show_tag_statistics())


def _categorize_tag(tag_name: str) -> str:
    """Categorize tag based on name."""
    nature_tags = {'landscape', 'mountain', 'tree', 'forest', 'ocean', 'sky', 'sunset', 'sunrise'}
    people_tags = {'person', 'people', 'family', 'child', 'adult', 'portrait'}
    object_tags = {'car', 'building', 'house', 'food', 'animal', 'dog', 'cat'}
    activity_tags = {'sports', 'running', 'swimming', 'hiking', 'travel', 'vacation'}
    
    if tag_name in nature_tags:
        return 'nature'
    elif tag_name in people_tags:
        return 'people'
    elif tag_name in object_tags:
        return 'objects'
    elif tag_name in activity_tags:
        return 'activities'
    else:
        return 'general'




# Add advanced commands as a subgroup
main.add_command(advanced_cli)


def cleanup_database():
    """Cleanup database connections on exit."""
    try:
        # Only try to cleanup if there's an event loop running
        try:
            loop = asyncio.get_running_loop()
            if loop and not loop.is_closed():
                db_engine = get_database_engine()
                asyncio.create_task(db_engine.close())
        except RuntimeError:
            # No running event loop, safe to ignore
            pass
    except Exception:
        # Ignore cleanup errors to prevent exit issues
        pass


if __name__ == '__main__':
    import atexit
    atexit.register(cleanup_database)
    main()