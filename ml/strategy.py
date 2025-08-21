def recovery_action_by_score(risk_score: float) -> str:
    """
    Rules:
    - risk_score > 0.75 -> Immediate legal action (High risk)
    - 0.50 <= risk_score <= 0.75 -> Settlement offers and repayment plans (Moderate risk)
    - risk_score < 0.50 -> Automated reminders (Low risk)
    """
    if risk_score > 0.75:
        return "Immediate Legal Action"
    if risk_score >= 0.50:
        return "Settlement Offer / Repayment Plan"
    return "Automated Reminders"


def apply_strategies(df_with_scores):
    df = df_with_scores.copy()
    df["recommended_action"] = df["risk_score"].apply(recovery_action_by_score)
    return df
