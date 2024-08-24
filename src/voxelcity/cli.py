"""Console script for voxelcity."""
import typer
from rich.console import Console
from voxelcity import generate_model

app = typer.Typer()
console = Console()

@app.command()
def main(city_name: str = typer.Argument(..., help="Name of the city to model"),
         output_format: str = typer.Option("obj", help="Output format for the 3D model")):
    """Generate a 3D city model."""
    generate_model(city_name, output_format)

if __name__ == "__main__":
    app()
