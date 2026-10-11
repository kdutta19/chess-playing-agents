"""Tests for the Claude agent loop, using a fake client instead of the real API."""

from types import SimpleNamespace

from chess_agents.agent import ClaudeAgent
from chess_agents.config import GameLimits
from chess_agents.tools import ToolResult


# ----- helpers that imitate the shape of Claude API responses ------------------

def text(content):
    return SimpleNamespace(type="text", text=content)


def tool_use(call_id, name, **arguments):
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments)


def response(*blocks):
    usage = SimpleNamespace(
        input_tokens=100, output_tokens=20,
        cache_read_input_tokens=None, cache_creation_input_tokens=None,
    )
    stop_reason = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, usage=usage)


class FakeClient:
    """Returns pre-written responses in order and remembers each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self   # so client.messages.create(...) works

    async def create(self, **request):
        # Copy the message list: the agent keeps appending to the original.
        self.requests.append({**request, "messages": list(request["messages"])})
        return self.responses.pop(0)


def make_agent(client, model="claude-opus-5-5"):
    return ClaudeAgent("Tester", "white", model, "balanced", "low", GameLimits(), client=client)


class ToolRecorder:
    """Stands in for the orchestrator: records tool calls and accepts any move."""

    def __init__(self):
        self.calls = []

    async def execute_tool(self, name, args):
        self.calls.append((name, args))
        if name == "make_move":
            return ToolResult("Move played.", ends_turn=True)
        return ToolResult("[]")


# ----- tests ---------------------------------------------------------------------

async def test_agent_runs_tools_then_moves():
    client = FakeClient([
        response(tool_use("call_1", "get_legal_moves")),
        response(text("I will open with the king's pawn."), tool_use("call_2", "make_move", move="e4", plan="develop")),
    ])
    agent, tools, said = make_agent(client), ToolRecorder(), []

    await agent.take_turn("Your turn.", tools.execute_tool, lambda kind, content: said.append((kind, content)))

    assert tools.calls == [("get_legal_moves", {}), ("make_move", {"move": "e4", "plan": "develop"})]
    assert said == [("agent_text", "I will open with the king's pawn.")]
    assert len(client.requests) == 2   # no extra API call after the move
    assert agent.usage.input_tokens == 200

    # The second request must carry the tool result, tied to the first call's id.
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "call_1"


async def test_agent_is_reminded_when_it_forgets_to_move():
    client = FakeClient([
        response(text("e4 looks best.")),   # talks but calls no tool
        response(tool_use("call_1", "make_move", move="e4", plan="")),
    ])
    agent, tools = make_agent(client), ToolRecorder()

    await agent.take_turn("Your turn.", tools.execute_tool, lambda kind, content: None)

    assert tools.calls == [("make_move", {"move": "e4", "plan": ""})]
    assert "make_move" in client.requests[1]["messages"][-1]["content"]


async def test_agent_gives_up_after_repeated_reminders():
    client = FakeClient([response(text("Thinking...")) for _ in range(3)])
    agent, tools = make_agent(client), ToolRecorder()

    await agent.take_turn("Your turn.", tools.execute_tool, lambda kind, content: None)

    assert tools.calls == []           # it never moved; the orchestrator will forfeit it
    assert client.responses == []      # the first try plus two reminders


async def test_request_settings_depend_on_the_model():
    move = [response(tool_use("call_1", "make_move", move="e4", plan=""))]

    opus = FakeClient(move.copy())
    await make_agent(opus).take_turn("Go.", ToolRecorder().execute_tool, lambda kind, content: None)
    assert opus.requests[0]["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert opus.requests[0]["output_config"] == {"effort": "low"}

    # Haiku 4.5 rejects both settings, so they must be left out.
    haiku = FakeClient(move.copy())
    await make_agent(haiku, "claude-haiku-4-5").take_turn("Go.", ToolRecorder().execute_tool, lambda kind, content: None)
    assert "thinking" not in haiku.requests[0]
    assert "output_config" not in haiku.requests[0]
