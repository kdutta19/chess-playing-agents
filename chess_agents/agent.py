"""A chess player.

ClaudeAgent is the heart of the project: a hand-written "agent loop". For
one turn it repeatedly

    1. sends the conversation to Claude,
    2. runs whatever tools Claude asked for,
    3. sends the tool results back,

until Claude plays a move. Nothing here knows the rules of chess or whose
turn it is; the agent only talks to Claude and to the `execute_tool`
function the orchestrator hands it.

RandomAgent has the same interface but picks moves at random. It shows that
anything with a `take_turn` method can sit at the board, and it lets the
project run without an API key.
"""

import asyncio
import json
import random
from dataclasses import dataclass
from typing import Awaitable, Callable

import anthropic

from .config import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    MAX_TOKENS,
    MODELS,
    PERSONAS,
    RANDOM_MODEL,
    GameLimits,
)
from .tools import TOOL_SCHEMAS, ToolResult

# The orchestrator gives the agent these two functions for each turn.
ExecuteTool = Callable[[str, dict], Awaitable[ToolResult]]   # run a tool, get its result
EmitText = Callable[[str, str], None]                        # publish ("agent_text" | "agent_thinking", text)

# Claude cannot be forced to call a tool on the newest models, so if it ends
# its reply without moving we remind it. After this many reminders we give up
# and the orchestrator forfeits the player.
MAX_REMINDERS = 2


class AgentError(Exception):
    """The agent could not play for a reason that is not its fault (API or credentials)."""


@dataclass
class Usage:
    """Running token totals for one player across the whole game."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, usage) -> None:
        """Add the usage reported by one API response."""
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0
        self.cache_read_tokens += usage.cache_read_input_tokens or 0
        self.cache_write_tokens += usage.cache_creation_input_tokens or 0

    def cost(self, model: str) -> float:
        """Estimated cost in US dollars at the prices in config.py."""
        info = MODELS[model]
        input_cost = (
            self.input_tokens
            + self.cache_read_tokens * CACHE_READ_MULTIPLIER
            + self.cache_write_tokens * CACHE_WRITE_MULTIPLIER
        ) * info.input_price
        output_cost = self.output_tokens * info.output_price
        return (input_cost + output_cost) / 1_000_000

    def as_dict(self, model: str) -> dict:
        return {
            "input_tokens": self.input_tokens + self.cache_read_tokens + self.cache_write_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost(model), 4),
        }


def build_system_prompt(color: str, persona: str, limits: GameLimits) -> str:
    """The standing instructions for a player. They never change during a game."""
    return f"""You are a chess player in a match against another AI. You play {color}.

Style: {PERSONAS[persona]}

How a turn works:
- At the start of your turn you are given the position, the moves so far and the note you left yourself last turn.
- You can call tools to look at the board, list the legal moves and get facts about the position.
- You finish your turn by calling make_move once. Until you do, your turn is not over.
- You may make at most {limits.tool_budget} tool calls per turn, and {limits.max_illegal_moves} illegal move attempts in one turn lose you the game, so check the legal moves when you are unsure.

You have no memory of earlier turns apart from the move list and your own note, so write a useful plan in make_move.

Spectators read what you write. Before you move, explain your choice in one or two plain sentences. Keep it short."""


class ClaudeAgent:
    """A player whose moves are chosen by a Claude model."""

    def __init__(
        self,
        name: str,
        color: str,
        model: str,
        persona: str,
        effort: str,
        limits: GameLimits,
        client: anthropic.AsyncAnthropic | None = None,
    ):
        self.name = name
        self.color = color
        self.model = model
        self.persona = persona
        self.effort = effort
        self.usage = Usage()
        self.system_prompt = build_system_prompt(color, persona, limits)
        # AsyncAnthropic() reads the API key from the ANTHROPIC_API_KEY
        # environment variable. Tests pass in a fake client instead.
        self.client = client or anthropic.AsyncAnthropic()

    async def take_turn(self, turn_prompt: str, execute_tool: ExecuteTool, emit: EmitText) -> None:
        """Play one turn. Returns once a move is accepted, or once the agent gives up."""
        # Each turn starts a fresh conversation. The prompt already holds
        # everything the model needs (position, history, its own note), so
        # we do not resend earlier turns. This keeps every turn about the
        # same size no matter how long the game gets.
        messages = [{"role": "user", "content": turn_prompt}]
        reminders = 0

        while True:
            response = await self._call_claude(messages)
            self.usage.add(response.usage)

            if response.stop_reason == "refusal":
                raise AgentError(f"{self.model} declined to respond")

            # Publish what the model said so the web page can show it.
            for block in response.content:
                if block.type == "thinking" and block.thinking:
                    emit("agent_thinking", block.thinking)
                elif block.type == "text" and block.text.strip():
                    emit("agent_text", block.text.strip())

            # Keep the model's full reply in the conversation. The API needs
            # the tool_use blocks (and any thinking blocks) sent back unchanged.
            if response.content:
                messages.append({"role": "assistant", "content": response.content})

            tool_calls = [block for block in response.content if block.type == "tool_use"]

            if not tool_calls:
                # The model stopped talking without playing a move.
                reminders += 1
                if reminders > MAX_REMINDERS:
                    return
                messages.append({
                    "role": "user",
                    "content": "You have not moved yet. Call make_move to play your move.",
                })
                continue

            # Run every tool the model asked for. All the results go back
            # together in a single user message, as the API expects.
            results = []
            for call in tool_calls:
                result = await execute_tool(call.name, call.input)
                if result.ends_turn:
                    return  # the move was accepted; nothing more to send
                results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,  # ties the result to the request
                    "content": result.content,
                    "is_error": result.is_error,
                })
            messages.append({"role": "user", "content": results})

    async def _call_claude(self, messages: list):
        """Send one request to Claude, turning API failures into AgentError."""
        info = MODELS[self.model]
        request = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": self.system_prompt,
            "tools": TOOL_SCHEMAS,
            "messages": messages,
            # Prompt caching: reuse the unchanged start of the request
            # (tools, system prompt, earlier messages of this turn) at a
            # tenth of the price. Requests below the model's minimum
            # cacheable size are simply not cached.
            "cache_control": {"type": "ephemeral"},
        }
        if info.adaptive_thinking:
            # Let the model decide how much to think. "summarized" returns a
            # readable summary of that thinking, which we show to spectators.
            request["thinking"] = {"type": "adaptive", "display": "summarized"}
        if info.supports_effort:
            request["output_config"] = {"effort": self.effort}

        # The SDK already retries rate limits, timeouts and server errors a
        # couple of times, so anything that reaches us here is final.
        try:
            return await self.client.messages.create(**request)
        except anthropic.AuthenticationError as error:
            raise AgentError("the API key was rejected; check ANTHROPIC_API_KEY") from error
        except anthropic.RateLimitError as error:
            raise AgentError("rate limited by the Claude API; try again shortly") from error
        except anthropic.APIStatusError as error:
            raise AgentError(f"Claude API error {error.status_code}: {error.message}") from error
        except anthropic.APIConnectionError as error:
            raise AgentError("could not reach the Claude API; check your connection") from error
        except TypeError as error:
            # The SDK raises TypeError when it finds no credentials at all.
            # Any other TypeError is a bug of ours and should not be hidden.
            if "authentication" not in str(error):
                raise
            raise AgentError("no Claude credentials found; put ANTHROPIC_API_KEY in a .env file") from error


class RandomAgent:
    """A player that picks a random legal move. It uses the same tools as ClaudeAgent."""

    def __init__(self, name: str, color: str, model: str, limits: GameLimits):
        self.name = name
        self.color = color
        self.model = model
        self.persona = "random"
        self.effort = "-"
        self.usage = Usage()
        self.delay = limits.random_delay

    async def take_turn(self, turn_prompt: str, execute_tool: ExecuteTool, emit: EmitText) -> None:
        await asyncio.sleep(self.delay)  # slow down so a person can follow the game
        legal = await execute_tool("get_legal_moves", {})
        move = random.choice(json.loads(legal.content))
        await execute_tool("make_move", {"move": move["uci"], "plan": "No plan: I move at random."})


def create_agent(name: str, color: str, model: str, persona: str, effort: str, limits: GameLimits):
    """Build the right kind of player for the chosen model."""
    if model == RANDOM_MODEL:
        return RandomAgent(name, color, model, limits)
    return ClaudeAgent(name, color, model, persona, effort, limits)
