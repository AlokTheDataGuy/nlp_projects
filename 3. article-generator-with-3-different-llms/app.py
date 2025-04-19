"""
Ollama Article Generator

A production-grade implementation of an article generator using Ollama LLMs.
"""

import os
import time
import json
import logging
import argparse
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("article_generator")


class ArticleStyle(str, Enum):
    """Enumeration of available article styles"""
    INFORMATIVE = "informative"
    PERSUASIVE = "persuasive"
    NARRATIVE = "narrative"
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"


class ArticleLength(str, Enum):
    """Enumeration of available article lengths"""
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


@dataclass
class ArticleRequest:
    """Data class for article generation requests"""
    topic: str
    style: ArticleStyle = ArticleStyle.INFORMATIVE
    length: ArticleLength = ArticleLength.MEDIUM


@dataclass
class ArticleResponse:
    """Data class for article generation responses"""
    article: str
    topic: str
    style: ArticleStyle
    length: ArticleLength
    model: str
    generation_time: float
    tokens: Optional[int] = None
    error: Optional[str] = None
    memory_usage: Optional[float] = None
    analysis: Optional[Dict[str, Any]] = None


@dataclass
class ModelConfig:
    """Data class for model configuration"""
    name: str
    system_prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text from the LLM"""
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM provider implementation"""

    def __init__(self, host: str = "http://localhost:11434"):
        """Initialize the Ollama provider"""
        self.host = host
        self._check_connection()

    def _check_connection(self) -> bool:
        """Check if Ollama server is running and available"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [model['name'] for model in response.json().get('models', [])]
                logger.info(f"Connected to Ollama. Available models: {', '.join(models)}")
                return True
            else:
                logger.error(f"Error connecting to Ollama: {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.error(f"Failed to connect to Ollama server at {self.host}: {str(e)}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            if response.status_code == 200:
                return [model['name'] for model in response.json().get('models', [])]
            logger.error(f"Failed to get models: {response.status_code}")
            return []
        except requests.RequestException as e:
            logger.error(f"Exception getting models: {str(e)}")
            return []

    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available in Ollama"""
        available_models = self.get_available_models()
        logger.info(f"Available models: {', '.join(available_models)}")
        return model_name in available_models

    def generate(self, prompt: str, system_prompt: str, model: str,
                 temperature: float = 0.7, max_tokens: int = 1024,
                 stream: bool = False, max_retries: int = 2) -> Dict[str, Any]:
        """Generate text using Ollama API with retry mechanism"""
        retries = 0
        while retries <= max_retries:
            try:
                logger.info(f"Generating text with model {model} (attempt {retries+1}/{max_retries+1})")
                response = requests.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "system": system_prompt,
                        "stream": stream,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        }
                    },
                    timeout=300  # Increased timeout for generation (5 minutes)
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Ollama API error: {response.status_code}, {response.text}")
                    error_msg = f"API error: {response.status_code}"
                    # Only retry on server errors (5xx)
                    if 500 <= response.status_code < 600 and retries < max_retries:
                        retries += 1
                        logger.info(f"Retrying due to server error ({retries}/{max_retries})")
                        time.sleep(2)  # Wait before retrying
                        continue
                    return {"error": error_msg, "message": response.text}

            except requests.RequestException as e:
                logger.error(f"Request to Ollama failed: {str(e)}")
                if retries < max_retries:
                    retries += 1
                    logger.info(f"Retrying after connection error ({retries}/{max_retries})")
                    time.sleep(2)  # Wait before retrying
                    continue
                return {"error": f"Request failed after {max_retries+1} attempts: {str(e)}"}


class ArticleGenerator:
    """Article generator using LLM providers"""

    # Default model configurations
    DEFAULT_MODELS = {
        "mistral": ModelConfig(
            name="mistral:7b",
            system_prompt="You are a professional writer that creates well-structured, engaging articles."
        ),
        "qwen2.5": ModelConfig(
            name="qwen2.5:7b",
            system_prompt="You are a professional writer that creates well-structured, engaging articles."
        ),
        "llama3.1": ModelConfig(
            name="llama3.1:8b",
            system_prompt="You are a professional writer that creates well-structured, engaging articles."
        ),
    }

    def __init__(self, provider: LLMProvider, config_path: Optional[str] = None):
        """Initialize the article generator with an LLM provider"""
        self.provider = provider
        self.models = self._load_config(config_path) if config_path else self.DEFAULT_MODELS

    def _load_config(self, config_path: str) -> Dict[str, ModelConfig]:
        """Load model configurations from a JSON file"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            models = {}
            for model_id, config in config_data.items():
                models[model_id] = ModelConfig(**config)

            logger.info(f"Loaded configuration for {len(models)} models from {config_path}")
            return models
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            logger.info("Using default model configurations")
            return self.DEFAULT_MODELS

    def save_config(self, config_path: str) -> bool:
        """Save current model configurations to a JSON file"""
        try:
            config_data = {model_id: asdict(config) for model_id, config in self.models.items()}
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {str(e)}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of configured models"""
        return list(self.models.keys())

    def _format_article_prompt(self, request: ArticleRequest) -> str:
        """Format the prompt for article generation"""
        return f"""Please write an {request.style.value} article about {request.topic}.
The article should be {request.length.value} in length and well-structured with clear paragraphs.
Include a compelling headline and organize the content in a coherent manner.
"""

    def generate_article(self, model_id: str, request: ArticleRequest) -> ArticleResponse:
        """Generate an article based on the request"""
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found in configuration")
            return ArticleResponse(
                article="",
                topic=request.topic,
                style=request.style,
                length=request.length,
                model=model_id,
                generation_time=0,
                error=f"Model {model_id} not configured"
            )

        # Check if the model is available in Ollama
        model_config = self.models[model_id]
        if not self.provider.is_model_available(model_config.name):
            logger.error(f"Model {model_config.name} not available in Ollama")
            return ArticleResponse(
                article="",
                topic=request.topic,
                style=request.style,
                length=request.length,
                model=model_id,
                generation_time=0,
                error=f"Model {model_config.name} not available in Ollama. Try running 'ollama pull {model_config.name}'"
            )
        prompt = self._format_article_prompt(request)

        # Measure generation time
        start_time = time.time()

        # Call the LLM provider
        result = self.provider.generate(
            prompt=prompt,
            system_prompt=model_config.system_prompt,
            model=model_config.name,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens
        )

        # Calculate generation time
        generation_time = time.time() - start_time

        # Check for errors
        if "error" in result:
            logger.error(f"Error generating article with {model_id}: {result['error']}")
            return ArticleResponse(
                article="",
                topic=request.topic,
                style=request.style,
                length=request.length,
                model=model_id,
                generation_time=generation_time,
                error=result.get("error", "Unknown error")
            )

        return ArticleResponse(
            article=result["response"],
            topic=request.topic,
            style=request.style,
            length=request.length,
            model=model_id,
            generation_time=generation_time,
            tokens=result.get("eval_count", None)
        )

    def _generate_with_memory_tracking(self, model_id: str, request: ArticleRequest) -> ArticleResponse:
        """Generate an article with memory usage tracking"""
        # Generate article
        response = self.generate_article(model_id, request)

        # Analyze article content
        if not response.error and response.article:
            try:
                from text_analysis import analyze_text
                analysis = analyze_text(response.article)
                response.analysis = analysis
            except ImportError:
                logger.warning("Text analysis module not available")

        return response

    def _generate_article_for_topic(self, model_id: str, topic: str, style: ArticleStyle, length: ArticleLength) -> ArticleResponse:
        """Generate an article for a specific topic"""
        request = ArticleRequest(topic=topic, style=style, length=length)
        return self._generate_with_memory_tracking(model_id, request)

    def benchmark(self, model_id: str, topics: List[str],
                  style: ArticleStyle = ArticleStyle.INFORMATIVE,
                  length: ArticleLength = ArticleLength.MEDIUM,
                  parallel: bool = False) -> Dict[str, Any]:
        """Run benchmark on a model with multiple topics"""
        results = []
        total_time = 0
        total_tokens = 0
        total_quality = 0
        errors = 0

        # Track overall benchmark time
        benchmark_start_time = time.time()

        if parallel and len(topics) > 1:
            # Use parallel processing for multiple topics
            try:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(topics), 3)) as executor:
                    # Create tasks for each topic
                    future_to_topic = {executor.submit(self._generate_article_for_topic, model_id, topic, style, length): topic
                                      for topic in topics}

                    # Process results as they complete
                    for future in concurrent.futures.as_completed(future_to_topic):
                        topic = future_to_topic[future]
                        try:
                            response = future.result()
                            if not response.error:
                                results.append(asdict(response))
                                total_time += response.generation_time
                                if response.tokens:
                                    total_tokens += response.tokens
                                if response.analysis:
                                    total_quality += response.analysis.get("quality_score", 0)
                            else:
                                errors += 1
                                logger.warning(f"Error with topic '{topic}': {response.error}")
                        except Exception as e:
                            errors += 1
                            logger.error(f"Exception processing topic '{topic}': {str(e)}")
            except ImportError:
                logger.warning("concurrent.futures not available, falling back to sequential processing")
                parallel = False

        if not parallel:
            # Sequential processing
            for topic in topics:
                response = self._generate_article_for_topic(model_id, topic, style, length)

                if not response.error:
                    results.append(asdict(response))
                    total_time += response.generation_time
                    if response.tokens:
                        total_tokens += response.tokens
                    if response.analysis:
                        total_quality += response.analysis.get("quality_score", 0)
                else:
                    errors += 1
                    logger.warning(f"Error with topic '{topic}': {response.error}")

        # Calculate total benchmark time
        benchmark_total_time = time.time() - benchmark_start_time

        if not results:
            return {"error": "All benchmark attempts failed", "model": model_id}

        return {
            "model": model_id,
            "results": results,
            "average_time": total_time / len(results),
            "total_time": total_time,
            "benchmark_time": benchmark_total_time,
            "average_tokens": total_tokens / len(results) if total_tokens > 0 else None,
            "average_quality": total_quality / len(results) if total_quality > 0 else None,
            "errors": errors,
            "success_rate": len(results) / (len(results) + errors)
        }


class ChatSession:
    """Manages an interactive chat session with the article generator"""

    def __init__(self, generator: ArticleGenerator):
        """Initialize a chat session with an article generator"""
        self.generator = generator
        self.current_model = None
        self.available_models = generator.get_available_models()
        logger.info(f"Chat session initialized. Available models: {', '.join(self.available_models)}")

    def _select_model(self) -> Optional[str]:
        """Prompt user to select a model"""
        if not self.available_models:
            print("No models available.")
            return None

        print(f"Available models: {', '.join(self.available_models)}")
        model = input("Select a model: ").strip().lower()

        if model in self.available_models:
            self.current_model = model
            print(f"Model {model} selected.")
            return model
        else:
            print(f"Invalid model. Please choose from: {', '.join(self.available_models)}")
            return None

    def _handle_command(self, command: str) -> bool:
        """Handle chat commands, returns True if should continue"""
        if command.lower() == "exit":
            print("Exiting chat session. Goodbye!")
            return False

        elif command.lower() == "help":
            self._show_help()
            return True

        elif command.lower().startswith("switch "):
            parts = command.split(" ", 1)
            if len(parts) > 1 and parts[1] in self.available_models:
                self.current_model = parts[1]
                print(f"Switched to {self.current_model}")
            else:
                print(f"Invalid model. Available models: {', '.join(self.available_models)}")
            return True

        elif command.lower().startswith("style "):
            parts = command.split(" ", 1)
            if len(parts) > 1:
                try:
                    self.current_style = ArticleStyle(parts[1].lower())
                    print(f"Article style set to {self.current_style.value}")
                except ValueError:
                    styles = [s.value for s in ArticleStyle]
                    print(f"Invalid style. Choose from: {', '.join(styles)}")
            return True

        elif command.lower().startswith("length "):
            parts = command.split(" ", 1)
            if len(parts) > 1:
                try:
                    self.current_length = ArticleLength(parts[1].lower())
                    print(f"Article length set to {self.current_length.value}")
                except ValueError:
                    lengths = [l.value for l in ArticleLength]
                    print(f"Invalid length. Choose from: {', '.join(lengths)}")
            return True

        return None  # Not a command

    def _show_help(self):
        """Show help information"""
        print("\nCommands:")
        print("  help                  - Show this help message")
        print("  exit                  - Exit the chat session")
        print("  switch <model>        - Switch to a different model")
        print("  style <style>         - Set article style (informative, persuasive, narrative, technical, conversational)")
        print("  length <length>       - Set article length (short, medium, long)")
        print("\nOr simply type a topic to generate an article.")

    def start(self):
        """Start an interactive chat session"""
        print("Welcome to the Article Generator Chat!")
        print("Type 'help' for available commands.")

        # Set defaults
        self.current_style = ArticleStyle.INFORMATIVE
        self.current_length = ArticleLength.MEDIUM

        # Select initial model if none selected
        if not self.current_model and not self._select_model():
            print("No model selected. Exiting.")
            return

        # Main chat loop
        while True:
            user_input = input(f"\n[{self.current_model}] >>> ").strip()

            if not user_input:
                continue

            # Check if input is a command
            command_result = self._handle_command(user_input)
            if command_result is not None:  # It was a command
                if not command_result:  # Should exit
                    break
                continue

            # Treat input as an article topic
            print(f"Generating {self.current_length.value} {self.current_style.value} article about '{user_input}'...")

            request = ArticleRequest(
                topic=user_input,
                style=self.current_style,
                length=self.current_length
            )

            try:
                response = self.generator.generate_article(self.current_model, request)

                if not response.error:
                    print("\n" + "="*50 + "\n")
                    print(response.article)
                    print("\n" + "="*50 + "\n")
                    print(f"Generation time: {response.generation_time:.2f} seconds")
                    if response.tokens:
                        print(f"Tokens generated: {response.tokens}")
                else:
                    print(f"Error: {response.error}")
            except Exception as e:
                logger.exception("Error during article generation")
                print(f"An error occurred: {str(e)}")


def get_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(description="Ollama Article Generator")

    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat session")
    chat_parser.add_argument("--host", default="http://localhost:11434",
                          help="Ollama API host address")
    chat_parser.add_argument("--model", help="Initial model to use")
    chat_parser.add_argument("--config", help="Path to model configuration file")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--host", default="http://localhost:11434",
                           help="Ollama API host address")
    bench_parser.add_argument("--models", nargs="+", help="Models to benchmark")
    bench_parser.add_argument("--topics", nargs="+",
                           default=["The Future of AI", "Climate Change", "Space Exploration"],
                           help="Topics to use for benchmarks")
    bench_parser.add_argument("--style", default="informative",
                           choices=[s.value for s in ArticleStyle],
                           help="Article style to use")
    bench_parser.add_argument("--length", default="medium",
                           choices=[l.value for l in ArticleLength],
                           help="Article length to use")
    bench_parser.add_argument("--parallel", action="store_true",
                           help="Run benchmarks in parallel")
    bench_parser.add_argument("--config", help="Path to model configuration file")
    bench_parser.add_argument("--output", help="Path to save benchmark results as JSON")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare models on the same topic")
    compare_parser.add_argument("--host", default="http://localhost:11434",
                             help="Ollama API host address")
    compare_parser.add_argument("--models", nargs="+", help="Models to compare")
    compare_parser.add_argument("--topic", required=True,
                             help="Topic to use for comparison")
    compare_parser.add_argument("--style", default="informative",
                             choices=[s.value for s in ArticleStyle],
                             help="Article style to use")
    compare_parser.add_argument("--length", default="medium",
                             choices=[l.value for l in ArticleLength],
                             help="Article length to use")
    compare_parser.add_argument("--config", help="Path to model configuration file")
    compare_parser.add_argument("--report", default="comparison_report.md",
                             help="Path to save comparison report")

    return parser


def main():
    """Main entry point"""
    parser = get_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize Ollama provider
    host = args.host
    provider = OllamaProvider(host=host)

    # Check if Ollama is available
    if not provider._check_connection():
        logger.error("Cannot connect to Ollama. Please make sure Ollama is running.")
        print("Please install Ollama: https://ollama.com/download")
        print("Then start the Ollama server: ollama serve")
        return

    # Initialize article generator
    generator = ArticleGenerator(provider, config_path=args.config if hasattr(args, 'config') else None)

    # Handle commands
    if args.command == "chat":
        session = ChatSession(generator)
        if hasattr(args, 'model') and args.model:
            session.current_model = args.model
        session.start()

    elif args.command == "benchmark":
        models_to_benchmark = args.models if args.models else generator.get_available_models()
        style = ArticleStyle(args.style)
        length = ArticleLength(args.length)
        parallel = args.parallel if hasattr(args, 'parallel') else False

        results = {}
        for model in models_to_benchmark:
            if model not in generator.get_available_models():
                logger.warning(f"Model {model} not configured, skipping")
                continue

            print(f"\nBenchmarking {model}...")
            benchmark_result = generator.benchmark(
                model, args.topics, style=style, length=length, parallel=parallel
            )
            results[model] = benchmark_result

            # Display results
            if "error" not in benchmark_result:
                print(f"  Success rate: {benchmark_result['success_rate'] * 100:.1f}%")
                print(f"  Average time: {benchmark_result['average_time']:.2f} seconds")
                if benchmark_result['average_tokens']:
                    print(f"  Average tokens: {benchmark_result['average_tokens']:.1f}")
                if benchmark_result.get('average_quality'):
                    print(f"  Average quality score: {benchmark_result['average_quality']:.1f}")
                if benchmark_result.get('benchmark_time'):
                    print(f"  Total benchmark time: {benchmark_result['benchmark_time']:.2f} seconds")
            else:
                print(f"  Error: {benchmark_result['error']}")

        # Save results if requested
        if hasattr(args, 'output') and args.output:
            try:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nBenchmark results saved to {args.output}")

                # Suggest using the comparison tool
                print("\nTo compare these results, run:")
                print(f"python model_comparison.py --input {args.output} --report comparison_report.md")
            except Exception as e:
                logger.error(f"Error saving benchmark results: {str(e)}")

    elif args.command == "compare":
        # Import model comparison functionality
        try:
            from model_comparison import analyze_benchmark_results, compare_models, generate_comparison_report
        except ImportError:
            logger.error("Could not import model_comparison module. Make sure model_comparison.py is in the same directory.")
            return

        models_to_compare = args.models if args.models else generator.get_available_models()
        style = ArticleStyle(args.style)
        length = ArticleLength(args.length)
        topic = args.topic

        print(f"\nComparing models on topic: '{topic}'...\n")

        # Generate articles with each model
        results = {}
        for model in models_to_compare:
            if model not in generator.get_available_models():
                logger.warning(f"Model {model} not configured, skipping")
                continue

            print(f"Generating with {model}...")
            request = ArticleRequest(topic=topic, style=style, length=length)
            response = generator._generate_with_memory_tracking(model, request)

            if not response.error:
                # Create a benchmark-like result structure with a single article
                results[model] = {
                    "model": model,
                    "results": [asdict(response)],
                    "average_time": response.generation_time,
                    "average_tokens": response.tokens,
                    "average_quality": response.analysis.get("quality_score", 0) if response.analysis else 0,
                    "success_rate": 1.0
                }
                print(f"  Done. Time: {response.generation_time:.2f}s, Tokens: {response.tokens or 'N/A'}")
            else:
                print(f"  Error: {response.error}")

        if not results:
            print("No successful generations to compare.")
            return

        # Analyze and compare results
        analyzed_results = analyze_benchmark_results(results)
        comparison = compare_models(analyzed_results)

        # Generate report
        if generate_comparison_report(comparison, args.report):
            print(f"\nComparison report saved to {args.report}")

        # Display summary
        if "best_model" in comparison and comparison["best_model"]:
            print(f"\nBest model for this topic: {comparison['best_model']}")

        # Ask if user wants to see the articles
        show_articles = input("\nWould you like to see the articles? (y/n): ").lower().startswith('y')
        if show_articles:
            for model, data in results.items():
                if "results" in data and data["results"]:
                    article = data["results"][0].get("article", "")
                    print(f"\n{'='*50}\n{model}\n{'='*50}\n")
                    print(article)
                    print("\n" + "-"*50)


if __name__ == "__main__":
    main()