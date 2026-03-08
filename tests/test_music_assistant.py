import mantra.interfaces.api.api_server as api_server
from mantra.music_assistant import handle_query


def test_music_assistant_prompt_parsing_and_endpoint():
    a = handle_query("find ambient piano tracks")
    b = handle_query("make playlist from track 42")

    assert a["intent"] == "search_text"
    assert b["intent"] == "playlist_generate"

    resp = api_server.assistant_query(api_server.AssistantQueryRequest(prompt="search lyrics for moonlight"))
    assert resp["response"]["intent"] in {"search_lyrics", "search_text"}
