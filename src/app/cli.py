"""Command Line Interface for the application."""
import argparse
import sys
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class CLIManager:
    """Command Line Interface manager."""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Python Application CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        self.setup_arguments()

    def setup_arguments(self):
        """Setup CLI arguments."""
        # Global options
        self.parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )

        self.parser.add_argument(
            '--config', '-c',
            type=str,
            default='config/config.yaml',
            help='Configuration file path'
        )

        # Subcommands
        subparsers = self.parser.add_subparsers(dest='command', help='Available commands')

        # Run command
        run_parser = subparsers.add_parser('run', help='Run the application')
        run_parser.add_argument(
            '--port', '-p',
            type=int,
            default=8000,
            help='Port to run on'
        )

        # Database commands
        db_parser = subparsers.add_parser('db', help='Database operations')
        db_subparsers = db_parser.add_subparsers(dest='db_command')

        db_subparsers.add_parser('init', help='Initialize database')
        db_subparsers.add_parser('migrate', help='Run database migrations')

        # Test command
        test_parser = subparsers.add_parser('test', help='Run tests')
        test_parser.add_argument(
            '--coverage',
            action='store_true',
            help='Run with coverage report'
        )

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command line arguments."""
        return self.parser.parse_args(args)

    def handle_command(self, args: argparse.Namespace):
        """Handle parsed command."""
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)

        if args.command == 'run':
            self.run_application(args)
        elif args.command == 'db':
            self.handle_db_command(args)
        elif args.command == 'test':
            self.run_tests(args)
        else:
            self.parser.print_help()

    def run_application(self, args: argparse.Namespace):
        """Run the main application."""
        logger.info(f"Starting application on port {args.port}")
        # Import and run your main application here
        print(f"Application would start on port {args.port}")

    def handle_db_command(self, args: argparse.Namespace):
        """Handle database commands."""
        if args.db_command == 'init':
            logger.info("Initializing database...")
            # Initialize database here
            print("Database initialized")
        elif args.db_command == 'migrate':
            logger.info("Running migrations...")
            # Run migrations here
            print("Migrations completed")

    def run_tests(self, args: argparse.Namespace):
        """Run tests."""
        import subprocess

        cmd = ['python', '-m', 'pytest', 'tests/']
        if args.coverage:
            cmd = ['coverage', 'run', '-m', 'pytest', 'tests/']

        try:
            subprocess.run(cmd, check=True)
            if args.coverage:
                subprocess.run(['coverage', 'report'], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Tests failed: {e}")
            sys.exit(1)

def main():
    """Main CLI entry point."""
    cli = CLIManager()
    args = cli.parse_args()
    cli.handle_command(args)

if __name__ == '__main__':
    main()
