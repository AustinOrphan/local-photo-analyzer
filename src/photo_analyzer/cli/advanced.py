"""Advanced CLI commands for Phase 4 features."""

import asyncio
from pathlib import Path
from typing import List, Optional
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text

from photo_analyzer.core.config import get_config
from photo_analyzer.core.logger import get_logger
from photo_analyzer.analyzer.advanced import AdvancedImageAnalyzer
from photo_analyzer.analyzer.duplicates import DuplicateDetector
from photo_analyzer.pipeline.batch import BatchProcessor, BatchConfig

console = Console()
logger = get_logger(__name__)


@click.group(name="advanced")
def advanced_cli():
    """Advanced analysis and processing commands."""
    pass


@advanced_cli.command()
@click.argument('image_path', type=click.Path(exists=True, path_type=Path))
@click.option('--ensemble', is_flag=True, help='Use ensemble of multiple models')
@click.option('--quality', is_flag=True, help='Include quality analysis')
@click.option('--duplicates', is_flag=True, help='Generate duplicate detection hash')
@click.option('--scene', is_flag=True, help='Include scene and composition analysis')
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Save results to JSON file')
def analyze(
    image_path: Path,
    ensemble: bool,
    quality: bool,
    duplicates: bool,
    scene: bool,
    output: Optional[Path]
):
    """Perform advanced analysis on a single image."""
    async def run_analysis():
        analyzer = AdvancedImageAnalyzer()
        
        with console.status(f"[bold green]Analyzing {image_path.name}..."):
            try:
                result = await analyzer.analyze_image_advanced(
                    image_path,
                    use_ensemble=ensemble,
                    quality_analysis=quality,
                    duplicate_detection=duplicates,
                    scene_analysis=scene
                )
                
                # Display results
                _display_analysis_results(result, image_path)
                
                # Save to file if requested
                if output:
                    _save_analysis_results(result, output)
                
            except Exception as e:
                console.print(f"[red]Analysis failed: {e}")
                raise click.Abort()
    
    asyncio.run(run_analysis())


@advanced_cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--types', multiple=True, default=['exact', 'near'], 
              help='Detection types (exact, near, similar)')
@click.option('--threshold', type=float, default=0.8, help='Similarity threshold')
@click.option('--suggest-resolution', is_flag=True, help='Suggest duplicate resolution')
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Save results to JSON file')
def find_duplicates(
    directory: Path,
    types: tuple,
    threshold: float,
    suggest_resolution: bool,
    output: Optional[Path]
):
    """Find duplicate images in a directory."""
    async def run_detection():
        detector = DuplicateDetector()
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif', '.webp', '.heic'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(directory.glob(f"**/*{ext}"))
            image_files.extend(directory.glob(f"**/*{ext.upper()}"))
        
        if not image_files:
            console.print(f"[yellow]No image files found in {directory}")
            return
        
        console.print(f"[blue]Found {len(image_files)} images to analyze")
        
        # Create mock Photo objects for testing
        # In real implementation, these would come from database
        photos = []  # Placeholder
        
        with console.status("[bold green]Detecting duplicates..."):
            try:
                duplicate_groups = await detector.detect_duplicates(
                    photos,
                    detection_types=list(types)
                )
                
                # Display results
                _display_duplicate_results(duplicate_groups, suggest_resolution, detector, photos)
                
                # Save to file if requested
                if output:
                    _save_duplicate_results(duplicate_groups, output)
                
            except Exception as e:
                console.print(f"[red]Duplicate detection failed: {e}")
                raise click.Abort()
    
    asyncio.run(run_detection())


@advanced_cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--batch-size', type=int, default=10, help='Number of images to process concurrently')
@click.option('--ensemble', is_flag=True, help='Use ensemble analysis')
@click.option('--output-dir', type=click.Path(path_type=Path), help='Directory to save results')
def batch_analyze(
    directory: Path,
    batch_size: int,
    ensemble: bool,
    output_dir: Optional[Path]
):
    """Perform batch analysis on all images in a directory."""
    async def run_batch():
        processor = BatchProcessor()
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif', '.webp', '.heic'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(directory.glob(f"**/*{ext}"))
            image_files.extend(directory.glob(f"**/*{ext.upper()}"))
        
        if not image_files:
            console.print(f"[yellow]No image files found in {directory}")
            return
        
        console.print(f"[blue]Starting batch analysis of {len(image_files)} images")
        
        # Create photo IDs (placeholder)
        photo_ids = [str(i) for i in range(len(image_files))]
        
        # Configure batch processing
        batch_config = BatchConfig(
            max_concurrent=batch_size,
            retry_attempts=2,
            progress_callback=_batch_progress_callback
        )
        
        try:
            # Start batch analysis
            batch_id = await processor.analyze_photos_batch(
                photo_ids,
                model=None if ensemble else "llava",
                batch_config=batch_config
            )
            
            console.print(f"[green]Batch analysis started with ID: {batch_id}")
            
            # Monitor progress
            await _monitor_batch_progress(processor, batch_id)
            
        except Exception as e:
            console.print(f"[red]Batch analysis failed: {e}")
            raise click.Abort()
    
    asyncio.run(run_batch())


@advanced_cli.command()
@click.argument('image_path', type=click.Path(exists=True, path_type=Path))
def quality_check(image_path: Path):
    """Analyze technical quality of an image."""
    async def run_quality_check():
        analyzer = AdvancedImageAnalyzer()
        
        with console.status(f"[bold green]Analyzing quality of {image_path.name}..."):
            try:
                quality_metrics = await analyzer._analyze_image_quality(image_path)
                scene_data = await analyzer._analyze_scene_and_colors(image_path)
                
                # Display quality results
                _display_quality_results(quality_metrics, scene_data, image_path)
                
            except Exception as e:
                console.print(f"[red]Quality analysis failed: {e}")
                raise click.Abort()
    
    asyncio.run(run_quality_check())


def _display_analysis_results(result, image_path: Path):
    """Display advanced analysis results in a formatted table."""
    console.print(Panel(
        f"[bold blue]Advanced Analysis Results for {image_path.name}",
        expand=False
    ))
    
    # Basic info table
    basic_table = Table(title="Analysis Summary")
    basic_table.add_column("Metric", style="cyan")
    basic_table.add_column("Value", style="green")
    
    basic_table.add_row("Description", result.description[:100] + "..." if len(result.description) > 100 else result.description)
    basic_table.add_row("Confidence Score", f"{result.confidence_score:.2%}")
    basic_table.add_row("Model Consensus", f"{result.model_consensus:.2%}")
    basic_table.add_row("Suggested Filename", result.suggested_filename)
    basic_table.add_row("Tags", ", ".join(result.tags[:5]) + ("..." if len(result.tags) > 5 else ""))
    
    console.print(basic_table)
    
    # Quality metrics if available
    if result.image_quality:
        quality_table = Table(title="Quality Metrics")
        quality_table.add_column("Metric", style="cyan")
        quality_table.add_column("Value", style="green")
        
        for metric, value in result.image_quality.items():
            if isinstance(value, float):
                quality_table.add_row(metric.replace('_', ' ').title(), f"{value:.3f}")
            else:
                quality_table.add_row(metric.replace('_', ' ').title(), str(value))
        
        console.print(quality_table)
    
    # Model metadata
    if result.metadata.get('models_used'):
        models_text = Text("Models Used: ", style="bold cyan")
        models_text.append(", ".join(result.metadata['models_used']), style="green")
        console.print(models_text)


def _display_duplicate_results(duplicate_groups, suggest_resolution: bool, detector, photos):
    """Display duplicate detection results."""
    if not duplicate_groups:
        console.print("[green]No duplicates found!")
        return
    
    console.print(Panel(
        f"[bold red]Found {len(duplicate_groups)} duplicate groups",
        expand=False
    ))
    
    for i, group in enumerate(duplicate_groups, 1):
        group_table = Table(title=f"Duplicate Group {i}")
        group_table.add_column("Property", style="cyan")
        group_table.add_column("Value", style="green")
        
        group_table.add_row("Type", group.duplicate_type)
        group_table.add_row("Similarity Score", f"{group.similarity_score:.2%}")
        group_table.add_row("Detection Method", group.detection_method)
        group_table.add_row("Photos", str(len(group.photo_ids)))
        group_table.add_row("Representative", group.representative_id)
        
        console.print(group_table)
        
        if suggest_resolution:
            # This would show resolution suggestions
            console.print(f"[yellow]Suggested action: Keep {group.representative_id}, review others")


def _display_quality_results(quality_metrics, scene_data, image_path: Path):
    """Display quality analysis results."""
    console.print(Panel(
        f"[bold blue]Quality Analysis for {image_path.name}",
        expand=False
    ))
    
    # Quality metrics table
    if quality_metrics:
        quality_table = Table(title="Technical Quality")
        quality_table.add_column("Metric", style="cyan")
        quality_table.add_column("Value", style="green")
        quality_table.add_column("Assessment", style="yellow")
        
        for metric, value in quality_metrics.items():
            assessment = _assess_quality_metric(metric, value)
            if isinstance(value, float):
                quality_table.add_row(
                    metric.replace('_', ' ').title(),
                    f"{value:.3f}",
                    assessment
                )
            else:
                quality_table.add_row(
                    metric.replace('_', ' ').title(),
                    str(value),
                    assessment
                )
        
        console.print(quality_table)
    
    # Scene analysis
    if scene_data:
        if 'dominant_colors' in scene_data:
            console.print("\n[bold cyan]Dominant Colors:")
            for color in scene_data['dominant_colors'][:5]:
                console.print(f"  {color['hex']} ({color['percentage']:.1f}%)")


def _assess_quality_metric(metric: str, value) -> str:
    """Assess quality metric and return human-readable assessment."""
    if metric == 'sharpness':
        if value > 0.7:
            return "Excellent"
        elif value > 0.4:
            return "Good"
        elif value > 0.2:
            return "Fair"
        else:
            return "Poor"
    elif metric == 'brightness':
        if 0.2 <= value <= 0.8:
            return "Good"
        elif value < 0.2:
            return "Too Dark"
        else:
            return "Too Bright"
    elif metric == 'contrast':
        if value > 0.3:
            return "Good"
        elif value > 0.15:
            return "Fair"
        else:
            return "Low"
    else:
        return "N/A"


async def _batch_progress_callback(batch_operation):
    """Callback for batch progress updates."""
    # This would be called during batch processing
    pass


async def _monitor_batch_progress(processor: BatchProcessor, batch_id: str):
    """Monitor and display batch processing progress."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[green]Processing batch...", total=100)
        
        while True:
            batch_operation = processor.get_batch_status(batch_id)
            if not batch_operation:
                break
            
            progress.update(task, completed=batch_operation.progress)
            
            if batch_operation.status.value in ['completed', 'failed', 'cancelled']:
                break
            
            await asyncio.sleep(1)
        
        if batch_operation:
            if batch_operation.status.value == 'completed':
                console.print(f"[green]Batch completed successfully!")
                console.print(f"Processed: {batch_operation.completed_items}/{batch_operation.total_items}")
                if batch_operation.failed_items > 0:
                    console.print(f"[yellow]Failed: {batch_operation.failed_items}")
            else:
                console.print(f"[red]Batch {batch_operation.status.value}")


def _save_analysis_results(result, output_path: Path):
    """Save analysis results to JSON file."""
    import json
    
    # Convert result to dictionary
    result_dict = {
        'description': result.description,
        'tags': result.tags,
        'suggested_filename': result.suggested_filename,
        'confidence_score': result.confidence_score,
        'model_consensus': result.model_consensus,
        'duplicate_hash': result.duplicate_hash,
        'image_quality': result.image_quality,
        'scene_analysis': result.scene_analysis,
        'color_analysis': result.color_analysis,
        'metadata': result.metadata
    }
    
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    
    console.print(f"[green]Results saved to {output_path}")


def _save_duplicate_results(duplicate_groups, output_path: Path):
    """Save duplicate detection results to JSON file."""
    import json
    
    groups_data = []
    for group in duplicate_groups:
        groups_data.append({
            'representative_id': group.representative_id,
            'photo_ids': group.photo_ids,
            'similarity_score': group.similarity_score,
            'duplicate_type': group.duplicate_type,
            'detection_method': group.detection_method,
            'metadata': group.metadata
        })
    
    result_dict = {
        'total_groups': len(duplicate_groups),
        'total_duplicates': sum(len(group.photo_ids) for group in duplicate_groups),
        'groups': groups_data
    }
    
    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    
    console.print(f"[green]Duplicate results saved to {output_path}")


if __name__ == "__main__":
    advanced_cli()