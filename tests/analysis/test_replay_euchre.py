"""Tests for replay_euchre_match_game (the god-view state reconstruction)."""

from __future__ import annotations

from gamesim.analysis import replay_euchre_match_game
from gamesim.core.agent import RandomAgent
from gamesim.core.types import AgentId
from gamesim.games.euchre import EuchreEngine, EuchreObservation, EuchreRules
from gamesim.games.euchre.actions import Action
from gamesim.recording import record_euchre_match
from gamesim.recording.euchre_match_log import EuchreMatchGameLog


def _sample_game(seed: int = 7, num_hands: int = 1) -> EuchreMatchGameLog:
    match = record_euchre_match(
        RandomAgent[EuchreObservation](seed=1),
        RandomAgent[EuchreObservation](seed=2),
        team_a_name="a",
        team_b_name="b",
        num_hands=num_hands,
        seed=seed,
    )
    return match.games[0]


def _direct_engine_replay(game: EuchreMatchGameLog) -> EuchreEngine:
    engine = EuchreEngine()
    engine.reset(seed=game.seed, dealer=game.dealer, rules=EuchreRules(game.stick_the_dealer))
    for agent, action in game.actions:
        engine.step(AgentId(agent), Action(action))
    return engine


def test_replay_length_is_actions_plus_one() -> None:
    game = _sample_game()

    snapshots = replay_euchre_match_game(game)

    assert len(snapshots) == len(game.actions) + 1


def test_first_snapshot_is_the_dealt_hand_before_any_action() -> None:
    game = _sample_game()

    snapshots = replay_euchre_match_game(game)
    first = snapshots[0]

    assert first.ply == 0
    assert first.last_action is None
    assert first.phase == "BID_ROUND_1"
    assert first.trump is None
    assert first.maker is None
    assert first.terminal is False
    assert sum(len(hand) for hand in first.hands) == 20  # 4 * 5 dealt cards
    assert all(0 <= card <= 23 for hand in first.hands for card in hand)


def test_final_snapshot_matches_direct_engine_replay() -> None:
    game = _sample_game()

    snapshots = replay_euchre_match_game(game)
    final = snapshots[-1]
    engine = _direct_engine_replay(game)
    final_obs = engine.observation(AgentId(0))

    assert final.terminal is True
    assert engine.is_terminal()
    assert final_obs.trump is not None
    assert final.points == final_obs.points
    assert final.scoring_team == final_obs.scoring_team
    assert final.alone == final_obs.alone
    assert final.maker == final_obs.maker
    assert final.trump == int(final_obs.trump)


def test_every_ply_total_cards_accounted_for_is_invariant() -> None:
    """20 cards are dealt (21 briefly, once the dealer picks up the upcard on an
    order-up, before their discard). Cards leave ``hands`` as they're played into
    a trick, so the invariant is hand-cards + cards-already-played, not hand-cards
    alone."""
    game = _sample_game()

    snapshots = replay_euchre_match_game(game)

    for snap in snapshots:
        active_players = 3 if snap.alone else 4
        cards_played_so_far = snap.trick_number * active_players + len(snap.current_trick)
        total_in_hand = sum(len(hand) for hand in snap.hands)
        assert total_in_hand + cards_played_so_far in (20, 21)


def test_last_action_label_is_phase_aware_for_discard_vs_play() -> None:
    """A card action means different things in DEALER_DISCARD vs TRICK_PLAY --
    confirm the precomputed label picks the right verb using phase-before-the-
    action, not phase-after."""
    # Search across a few seeds for a hand that goes through an order-up (so both
    # a discard and a later play of the same action-space kind both occur).
    found_discard = False
    found_play = False
    for seed in range(20):
        game = _sample_game(seed=seed)
        for snap in replay_euchre_match_game(game):
            if snap.last_action is None:
                continue
            if "discards" in snap.last_action.label:
                found_discard = True
            if "plays" in snap.last_action.label:
                found_play = True
        if found_discard and found_play:
            break

    assert found_discard, "fixture never exercised a discard across 20 seeds"
    assert found_play, "fixture never exercised a play across 20 seeds"


def test_snapshots_are_json_serializable() -> None:
    import json
    from dataclasses import asdict

    game = _sample_game()
    snapshots = replay_euchre_match_game(game)

    payload = json.dumps([asdict(s) for s in snapshots])
    assert isinstance(payload, str)
    round_tripped = json.loads(payload)
    assert len(round_tripped) == len(snapshots)
