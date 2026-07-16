"""UI rendering module for CheetahClaws."""
from .render import (
    C, clr, info, ok, warn, err, _truncate_err_global,
    render_diff, _has_diff,
    stream_text, stream_thinking, flush_response,
    _start_live,
    _TOOL_SPINNER_PHRASES, _DEBATE_SPINNER_PHRASES,
    _start_tool_spinner, _stop_tool_spinner, _change_spinner_phrase,
    print_tool_start, print_tool_end, _tool_desc,
    set_rich_live, set_stream_mode, auto_stream_mode, set_spinner_tips,
)
