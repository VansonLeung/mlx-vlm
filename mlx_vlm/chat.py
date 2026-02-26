import argparse
import os
import re
import sys
import time
from typing import Dict, List

import cv2
import mlx.core as mx
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from PIL import Image

from mlx_vlm import load
from mlx_vlm.generate import generate_step
from mlx_vlm.prompt_utils import get_message_json
from mlx_vlm.utils import load_image


class MLXVisionChat:
    def __init__(
        self,
        model_path: str = "mlx-community/idefics2-8b-chatty-4bit",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        verbose: bool = False,
    ):
        self.console = Console()
        self.verbose = verbose
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history: List[Dict] = []
        self.current_image = None

        with self.console.status("[bold green]Loading model..."):
            self.model, self.processor = load(model_path)

        rprint("[bold green]Model loaded successfully![/bold green]")
        self.print_help()

    def print_help(self) -> None:
        """Print available commands."""
        help_text = """
[bold yellow]Available Commands:[/bold yellow]
• /image <path> - Load a new image for discussion
    • /video <path> - Load a video (uses a sampled frame grid)
• /clear - Clear conversation history
• /help - Show this help message
• /exit - Exit the chat
• Any other input will be treated as a question or comment about the current image
        """
        rprint(Panel(help_text, title="Help", border_style="blue"))

    def process_image(self, image_path: str) -> bool:
        """Process an image and prepare it for the model. Returns True if successful."""
        try:
            if not os.path.exists(image_path):
                rprint(
                    f"[bold red]Error:[/bold red] Image file not found: {image_path}"
                )
                return False

            self.current_image = load_image(image_path)
            rprint(f"[bold blue]Loaded image:[/bold blue] {image_path}")
            return True
        except Exception as e:
            rprint(f"[bold red]Error loading image:[/bold red] {str(e)}")
            return False

    def process_video(self, video_path: str) -> bool:
        """Process a video by sampling frames and building a 2x2 grid."""
        try:
            if not os.path.exists(video_path):
                rprint(
                    f"[bold red]Error:[/bold red] Video file not found: {video_path}"
                )
                return False

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            sample_indices = [0]
            if total_frames > 1:
                sample_indices = [
                    0,
                    max(total_frames // 3, 0),
                    max((2 * total_frames) // 3, 0),
                    max(total_frames - 1, 0),
                ]

            sampled_frames = []
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok and frame is not None:
                    sampled_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cap.release()

            if len(sampled_frames) == 0:
                rprint(
                    f"[bold red]Error:[/bold red] Could not read frames from video: {video_path}"
                )
                return False

            if len(sampled_frames) == 1:
                combined = sampled_frames[0]
            else:
                min_h = min(f.shape[0] for f in sampled_frames)
                min_w = min(f.shape[1] for f in sampled_frames)
                sampled_frames = [
                    cv2.resize(f, (min_w, min_h), interpolation=cv2.INTER_AREA)
                    for f in sampled_frames[:4]
                ]
                while len(sampled_frames) < 4:
                    sampled_frames.append(sampled_frames[-1])

                top = cv2.hconcat(sampled_frames[:2])
                bottom = cv2.hconcat(sampled_frames[2:4])
                combined = cv2.vconcat([top, bottom])

            self.current_image = Image.fromarray(combined)
            rprint(
                f"[bold blue]Loaded video:[/bold blue] {video_path} [dim](using sampled frame grid)[/dim]"
            )
            return True
        except Exception as e:
            rprint(f"[bold red]Error loading video:[/bold red] {str(e)}")
            return False

    def add_to_history(self, role: str, text: str) -> None:
        """Add a message to the conversation history."""
        content = [{"type": "text", "text": text}]
        self.history.append({"role": role, "content": content})

    def generate_response(self) -> str:
        """Generate a response from the model based on the conversation history."""
        self._streaming_output = False
        if self.current_image is None:
            return "Please load an image or video first using /image or /video."

        supports_multimodal = hasattr(self.processor, "image_processor")
        if not supports_multimodal and str(self.model.config.model_type).lower() in {
            "minicpmo",
            "minicpm-o",
            "minicpm_o",
        }:
            return (
                "MiniCPM-o is currently running in text-only compatibility mode in mlx-vlm. "
                "Image/video understanding is not available yet for this checkpoint."
            )

        messages = []
        if str(self.model.config.model_type).lower() in {
            "minicpmo",
            "minicpm-o",
            "minicpm_o",
        }:
            messages.append(
                {
                    "role": "system",
                    "content": "Answer directly based on the visual input. Do not output <think> tags or chain-of-thought.",
                }
            )

        for i, message in enumerate(self.history):
            skip_token = True
            if i == len(self.history) - 1 and message["role"] == "user":
                skip_token = False
            if supports_multimodal:
                messages.append(
                    get_message_json(
                        self.model.config.model_type,
                        message["content"][0]["text"],
                        role=message["role"],
                        skip_image_token=skip_token,
                        num_images=1,
                    )
                )
            else:
                # Tokenizer-only fallback (e.g., MiniCPM-o compatibility mode)
                messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"][0]["text"],
                    }
                )

        text_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        if isinstance(text_prompt, list):
            if len(text_prompt) == 0:
                text_prompt = ""
            elif all(isinstance(x, int) for x in text_prompt):
                text_prompt = self.processor.decode(text_prompt)
            else:
                text_prompt = text_prompt[0]

        if not isinstance(text_prompt, str):
            text_prompt = str(text_prompt)

        if supports_multimodal:
            inputs = self.processor(
                text=[text_prompt],
                images=[self.current_image],
                padding=True,
                return_tensors="np",
            )
            pixel_values = mx.array(inputs["pixel_values"])
        else:
            inputs = self.processor(
                [text_prompt],
                padding=True,
                return_tensors="np",
            )
            pixel_values = None

        input_ids = mx.array(inputs["input_ids"])
        mask = mx.array(inputs["attention_mask"])

        detokenizer = self.processor.detokenizer
        detokenizer.reset()

        tic = time.perf_counter()

        generation_model = self.model.thinker if hasattr(self.model, "thinker") else self.model

        generator = generate_step(
            input_ids,
            generation_model,
            pixel_values,
            mask,
            temperature=self.temperature,
        )

        # Use print instead of rprint to avoid rich console's automatic newlines
        self._streaming_output = True
        rprint("[bold green]Assistant:[/bold green]", end=" ", flush=True)
        for (token, prob), n in zip(generator, range(self.max_tokens)):
            if n == 0:
                tic = time.perf_counter()

            eos_token_id = (
                self.processor.tokenizer.eos_token_id
                if hasattr(self.processor, "tokenizer")
                else self.processor.eos_token_id
            )
            if token == eos_token_id and n > 0:
                break

            detokenizer.add_token(token)

            if self.verbose:
                rprint(detokenizer.last_segment, end="", flush=True)

        detokenizer.finalize()
        text = detokenizer.text

        # Hide explicit reasoning traces if the model emits them.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if text.startswith("<think>"):
            # Handle truncated generations with no closing tag.
            split_idx = text.find("\n\n")
            if split_idx != -1:
                text = text[split_idx + 2 :].strip()
            else:
                text = ""

        return text

    def handle_command(self, command: str, args: str) -> bool:
        """Handle special commands. Returns True if should continue chat, False if should exit."""
        if command == "/exit":
            rprint("[bold yellow]Goodbye![/bold yellow]")
            return False
        elif command == "/help":
            self.print_help()
        elif command == "/clear":
            self.history.clear()
            rprint("[bold blue]Conversation history cleared.[/bold blue]")
        elif command == "/image":
            if not args:
                rprint("[bold red]Error:[/bold red] Please provide an image path")
                return True
            self.process_image(args.strip())
        elif command == "/video":
            if not args:
                rprint("[bold red]Error:[/bold red] Please provide a video path")
                return True
            self.process_video(args.strip())
        else:
            rprint(f"[bold red]Unknown command:[/bold red] {command}")
        return True

    def chat_loop(self) -> None:
        """Main chat loop for interaction."""
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()

                # Handle commands
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    command = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else ""
                    if not self.handle_command(command, args):
                        break
                    continue
                # Handle regular chat input
                if self.current_image is None:
                    rprint(
                        "[bold yellow]Please load an image or video first using /image or /video[/bold yellow]"
                    )
                    continue

                self.add_to_history("user", user_input)
                response = self.generate_response()

                if self.verbose and not getattr(self, "_streaming_output", False):
                    rprint(Panel(Markdown(response), border_style="yellow"))

                if not self.verbose:
                    rprint(Panel(Markdown(response), border_style="green"))

                # Remove the eos token from the response
                response = response.replace("<end_of_utterance>", "")

                self.add_to_history("assistant", response)

            except KeyboardInterrupt:
                rprint(
                    "\n[bold yellow]Interrupted by user. Type /exit to quit.[/bold yellow]"
                )
                continue
            except Exception as e:
                rprint(f"[bold red]Error:[/bold red] {str(e)}")
                continue


def main():
    parser = argparse.ArgumentParser(description="MLX Vision Chat CLI")
    parser.add_argument(
        "--model",
        default="mlx-community/idefics2-8b-chatty-4bit",
        help="Path to the model or model identifier",
    )
    parser.add_argument("--verbose", action="store_false", help="Enable verbose output")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for the model"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum number of new tokens to generate",
    )

    args = parser.parse_args()

    try:
        chat = MLXVisionChat(
            model_path=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
        )
        chat.chat_loop()
    except Exception as e:
        rprint(f"[bold red]Fatal error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
