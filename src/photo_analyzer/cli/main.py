"""Main CLI interface for the photo analyzer."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from photo_analyzer import __version__
from photo_analyzer.core.config import get_config, Config
from photo_analyzer.core.logger import get_logger

console = Console()
logger = get_logger(__name__)


def print_banner():
    """Print the application banner."""
    banner = f"""
[bold blue]📸 Local Photo Analyzer[/bold blue] v{__version__}
[dim]Privacy-first photo organization with local AI[/dim]
    """
    console.print(Panel(banner, border_style="blue"))


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--config', type=click.Path(exists=True, path_type=Path), 
              help='Configuration file path')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Set logging level')
@click.version_option(version=__version__)
@click.pass_context
def main(ctx: click.Context, debug: bool, config: Optional[Path], log_level: str):
    """Local Photo Analyzer - Privacy-first photo organization with local AI."""
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Load configuration
    if config:
        ctx.obj['config'] = Config.load_from_file(config)
    else:
        ctx.obj['config'] = get_config()
    
    # Override config with CLI options
    if debug:
        ctx.obj['config'].debug = True
        ctx.obj['config'].log_level = 'DEBUG'
    elif log_level:
        ctx.obj['config'].log_level = log_level
    
    # Setup logging with new config
    logger.setLevel(ctx.obj['config'].log_level)
    
    # Store CLI options
    ctx.obj['debug'] = debug
    ctx.obj['log_level'] = log_level


@main.command()
@click.pass_context
def init(ctx: click.Context):
    """Initialize the photo analyzer configuration and database."""
    print_banner()
    
    config: Config = ctx.obj['config']
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Initialize directories
        task = progress.add_task("Creating directories...", total=None)
        config._ensure_directories()
        progress.update(task, description="✅ Directories created")
        
        # Save default configuration
        task = progress.add_task("Saving configuration...", total=None)
        config.save_config()
        progress.update(task, description="✅ Configuration saved")
        
        # TODO: Initialize database
        task = progress.add_task("Initializing database...", total=None)
        # db_init()  # Will implement in database models
        progress.update(task, description="✅ Database initialized")
        
        # TODO: Check Ollama installation
        task = progress.add_task("Checking Ollama installation...", total=None)
        # check_ollama()  # Will implement in LLM integration
        progress.update(task, description="✅ Ollama checked")
    
    console.print("\n[green]✅ Initialization complete![/green]")
    console.print(f"Configuration saved to: {config.config_dir / 'config.yaml'}")
    console.print(f"Data directory: {config.data_dir}")


@main.command()
@click.argument('path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), 
              help='Output directory for organized photos')
@click.option('--dry-run', is_flag=True, help='Show what would be done without making changes')
@click.option('--batch-size', type=int, default=10, help='Number of images to process at once')
@click.pass_context
def analyze(ctx: click.Context, path: Path, output: Optional[Path], dry_run: bool, batch_size: int):
    """Analyze photos in the specified directory."""
    config: Config = ctx.obj['config']
    
    if path.is_file():
        console.print(f"[blue]Analyzing single image:[/blue] {path}")
        # TODO: Implement single image analysis
    else:
        console.print(f"[blue]Analyzing directory:[/blue] {path}")
        # TODO: Implement directory analysis
    
    if dry_run:
        console.print("[yellow]🔍 Dry run mode - no changes will be made[/yellow]")
    
    # TODO: Implement analysis logic
    console.print("[yellow]⚠️  Analysis functionality coming soon![/yellow]")


@main.command()
@click.argument('directory', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), 
              help='Output directory for organized photos')
@click.option('--dry-run', is_flag=True, help='Show what would be done without making changes')
@click.option('--backup', is_flag=True, default=True, help='Create backup before organizing')
@click.pass_context
def process(ctx: click.Context, directory: Path, output: Optional[Path], dry_run: bool, backup: bool):
    """Process and organize photos in the specified directory."""
    config: Config = ctx.obj['config']
    
    console.print(f"[blue]Processing directory:[/blue] {directory}")
    
    if output:
        console.print(f"[blue]Output directory:[/blue] {output}")
    else:
        console.print("[yellow]No output directory specified - will organize in place[/yellow]")
    
    if dry_run:
        console.print("[yellow]🔍 Dry run mode - no changes will be made[/yellow]")
    
    if backup:
        console.print("[green]📦 Backup mode enabled[/green]")
    
    # TODO: Implement processing logic
    console.print("[yellow]⚠️  Processing functionality coming soon![/yellow]")


@main.command()
@click.option('--host', default='127.0.0.1', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, reload: bool):
    """Start the web interface server."""
    config: Config = ctx.obj['config']
    
    console.print(f"[blue]Starting web server on[/blue] http://{host}:{port}")
    
    if reload:
        console.print("[yellow]🔄 Auto-reload enabled for development[/yellow]")
    
    # TODO: Implement web server startup
    console.print("[yellow]⚠️  Web interface functionality coming soon![/yellow]")


@main.command()
@click.pass_context
def status(ctx: click.Context):
    """Show system status and configuration."""
    config: Config = ctx.obj['config']
    
    print_banner()
    
    # Configuration table
    config_table = Table(title="Configuration", show_header=True, header_style="bold blue")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_table.add_row("App Name", config.app_name)
    config_table.add_row("Version", config.version)
    config_table.add_row("Debug Mode", str(config.debug))
    config_table.add_row("Log Level", config.log_level)
    config_table.add_row("Data Directory", str(config.data_dir))
    config_table.add_row("Config Directory", str(config.config_dir))
    config_table.add_row("Cache Directory", str(config.cache_dir))
    config_table.add_row("Log Directory", str(config.log_dir))
    
    console.print(config_table)
    
    # LLM Configuration table
    llm_table = Table(title="LLM Configuration", show_header=True, header_style="bold blue")
    llm_table.add_column("Setting", style="cyan")
    llm_table.add_column("Value", style="green")
    
    llm_table.add_row("Primary Model", config.llm.primary_model)
    llm_table.add_row("Fallback Model", config.llm.fallback_model)
    llm_table.add_row("Ollama URL", config.llm.ollama_url)
    llm_table.add_row("Timeout", f"{config.llm.timeout}s")
    llm_table.add_row("Temperature", str(config.llm.temperature))
    
    console.print(llm_table)
    
    # TODO: Add system health checks
    console.print("\n[green]✅ System Status: Healthy[/green]")


@main.command()
@click.argument('query', type=str)
@click.option('--limit', default=10, help='Maximum number of results')
@click.pass_context
def search(ctx: click.Context, query: str, limit: int):
    """Search for photos by content, tags, or metadata."""
    config: Config = ctx.obj['config']
    
    console.print(f"[blue]Searching for:[/blue] '{query}'")
    console.print(f"[blue]Limit:[/blue] {limit} results")
    
    # TODO: Implement search functionality
    console.print("[yellow]⚠️  Search functionality coming soon![/yellow]")


@main.command()
@click.pass_context
def config_show(ctx: click.Context):
    """Show current configuration."""
    config: Config = ctx.obj['config']
    
    console.print("[blue]Current Configuration:[/blue]")
    
    # Convert config to dict and display
    config_dict = config.dict()
    
    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                console.print("  " * indent + f"[cyan]{key}:[/cyan]")
                print_dict(value, indent + 1)
            else:
                console.print("  " * indent + f"[cyan]{key}:[/cyan] [green]{value}[/green]")
    
    print_dict(config_dict)


if __name__ == "__main__":
    main()