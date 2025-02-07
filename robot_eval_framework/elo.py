def update_elo_ratings(eloA: float, eloB: float, scoreA: float, k_factor: float = 32) -> (float, float):
    """
    Updates and returns the new ELO scores for A and B given their old ELO scores
    and a match result. scoreA is 1 if A wins, 0.5 if draw, 0 if B wins.
    """
    expectedA = 1 / (1 + 10 ** ((eloB - eloA) / 400))
    expectedB = 1 / (1 + 10 ** ((eloA - eloB) / 400))

    newA = eloA + k_factor * (scoreA - expectedA)
    newB = eloB + k_factor * ((1 - scoreA) - expectedB)
    return newA, newB
