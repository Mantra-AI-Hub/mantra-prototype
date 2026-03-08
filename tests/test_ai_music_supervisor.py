from mantra.ai_music_supervisor import AIMusicSupervisor


def test_ai_music_supervisor_status_and_cycle():
    supervisor = AIMusicSupervisor()
    status = supervisor.run_supervision_cycle(performance_score=0.4)
    assert "recommender" in status
    assert "experimentation" in status
    assert "metrics" in supervisor.status()
