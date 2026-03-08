from mantra.music_agent_system import MusicAgentSystem


def test_music_agent_system_runs_agents():
    system = MusicAgentSystem()
    discovery = system.run_discovery_agent([{"track_id": "t1"}])
    playlist = system.run_playlist_agent(discovery, length=1)
    trends = system.run_trend_agent([{"track_id": "t2"}])
    generated = system.run_generation_agent(["ambient rise"])
    status = system.status()
    assert playlist == ["t1"]
    assert trends
    assert generated[0]["status"] == "queued"
    assert "agents" in status

