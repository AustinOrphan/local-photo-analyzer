"""Main CLI interface for photo analyzer."""

import asyncio
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from ..core.config import get_config
from ..core.logger import setup_logging, get_logger
from ..database.engine import get_database_engine
from ..database.migrations import get_migration_manager
from ..pipeline.analyzer import PhotoAnalyzer
from ..pipeline.processor import PhotoProcessor
from ..pipeline.organizer import PhotoOrganizer

console = Console()
logger = get_logger(__name__)


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--config-file', type=click.Path(exists=True), help='Path to configuration file')
@click.pass_context
def main(ctx: click.Context, debug: bool, config_file: Optional[str]):
    """Photo Analyzer CLI - Secure local LLM-based photo analysis and organization."""
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
@click.option('--database-type', type=click.Choice(['sqlite', 'postgresql']), default='sqlite')
@click.option('--database-path', type=click.Path(), help='Database file path (for SQLite)')
@click.option('--reset', is_flag=True, help='Reset existing database')
@click.pass_context
def init(ctx: click.Context, database_type: str, database_path: Optional[str], reset: bool):
    """Initialize the photo analyzer database and configuration."""
    config = ctx.obj['config']
    
    console.print("[blue]Initializing Photo Analyzer...[/blue]")
    
    async def init_database():
        try:
            # Get database engine
            db_engine = get_database_engine()
            
            if reset:
                console.print("[yellow]Resetting database...[/yellow]")
                await db_engine.drop_all_tables()
            
            # Create tables
            console.print("[blue]Creating database tables...[/blue]")
            await db_engine.create_all_tables()
            
            # Run migrations
            console.print("[blue]Running migrations...[/blue]")
            migration_manager = get_migration_manager()
            await migration_manager.migrate_up()
            
            # Show migration status
            status = await migration_manager.status()
            console.print(f"[green]Database initialized successfully![/green]")
            console.print(f"Applied migrations: {status['applied_count']}")
            
        except Exception as e:
            console.print(f"[red]Database initialization failed: {e}[/red]")
            raise
    
    # Run async initialization
    asyncio.run(init_database())
    console.print("[green]Photo Analyzer initialization complete![/green]")


@main.command()
@click.argument('paths', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--batch-size', default=5, help='Number of photos to process concurrently')
@click.option('--output-format', type=click.Choice(['json', 'table']), default='table')
@click.pass_context
def analyze(ctx: click.Context, paths: tuple, batch_size: int, output_format: str):
    """Analyze photos using local LLM."""
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
    
    async def analyze_photos():
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
        
        # Display results
        if output_format == 'table':
            display_analysis_table(results)
        else:
            import json
            console.print(json.dumps(results, indent=2, default=str))
    
    # Run analysis
    asyncio.run(analyze_photos())


@main.command()
@click.argument('paths', nargs=-1, required=True, type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--date-format', default='YYYY/MM/DD', help='Date directory format')
@click.option('--create-symlinks/--no-symlinks', default=True, help='Create categorical symlinks')
@click.option('--filename-strategy', type=click.Choice(['smart', 'preserve']), default='smart')
@click.option('--dry-run', is_flag=True, help='Show what would be done without making changes')
@click.option('--batch-size', default=5, help='Number of photos to process concurrently')
@click.pass_context
def organize(ctx: click.Context, paths: tuple, output_dir: str, date_format: str, 
            create_symlinks: bool, filename_strategy: str, dry_run: bool, batch_size: int):
    """Organize photos into date-based directory structure."""
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
@click.pass_context
def status(ctx: click.Context):
    """Show photo analyzer status and statistics."""
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
                health = await llm_client.health_check()
                llm_status = "✓ Connected" if health else "✗ Unavailable"
            except Exception:
                llm_status = "✗ Error"
            
            table.add_row("LLM Service", llm_status, config.llm.base_url)
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Failed to get status: {e}[/red]")
    
    asyncio.run(show_status())


@main.command()
@click.argument('query', required=True)
@click.option('--limit', default=10, help='Maximum number of results')
@click.option('--output-format', type=click.Choice(['json', 'table']), default='table')
@click.pass_context
def search(ctx: click.Context, query: str, limit: int, output_format: str):
    """Search photos by description, tags, or metadata."""
    config = ctx.obj['config']
    
    async def search_photos():
        from sqlalchemy import select, or_
        from ..database.session import get_async_db_session
        from ..models.photo import Photo, Tag
        
        async with get_async_db_session() as session:
            # Build search query
            stmt = select(Photo).where(
                or_(
                    Photo.description.contains(query),
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
@click.option('--check-health', is_flag=True, help='Check LLM service health')
@click.pass_context
def serve(ctx: click.Context, check_health: bool):
    """Start the photo analyzer web service (future implementation)."""
    console.print("[yellow]Web service not yet implemented[/yellow]")
    
    if check_health:
        async def check_llm_health():
            try:
                from ..analyzer.llm_client import OllamaClient
                config = ctx.obj['config']
                llm_client = OllamaClient(config)
                health = await llm_client.health_check()
                
                if health:
                    console.print("[green]✓ LLM service is healthy[/green]")
                else:
                    console.print("[red]✗ LLM service is not responding[/red]")
            except Exception as e:
                console.print(f"[red]✗ LLM health check failed: {e}[/red]")
        
        asyncio.run(check_llm_health())


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
        description = (photo.description[:40] + "...") if photo.description and len(photo.description) > 40 else (photo.description or "")
        tags = ", ".join([tag.name for tag in photo.tags]) if photo.tags else ""
        date_taken = photo.date_taken.strftime("%Y-%m-%d") if photo.date_taken else ""
        path = str(Path(photo.current_path).parent)
        
        table.add_row(filename, description, tags, date_taken, path)
    
    console.print(table)


if __name__ == '__main__':
    main()