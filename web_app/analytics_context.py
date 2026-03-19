import pandas as pd


def build_behavioral_summary(df: pd.DataFrame) -> dict:
    churned_df = df[df["Attrition_Flag"] == 1]
    retained_df = df[df["Attrition_Flag"] == 0]

    card_churn = (
        df.groupby("Card_Category")["Attrition_Flag"]
        .mean()
        .sort_values(ascending=False)
        .mul(100)
    )
    highest_risk_card = card_churn.index[0]
    highest_risk_card_rate = round(float(card_churn.iloc[0]), 2)

    return {
        "behavioral_patterns": {
            "avg_transaction_amount_churned": round(
                float(churned_df["Total_Trans_Amt"].mean()), 2
            ),
            "avg_transaction_amount_retained": round(
                float(retained_df["Total_Trans_Amt"].mean()), 2
            ),
            "avg_transaction_count_churned": round(
                float(churned_df["Total_Trans_Ct"].mean()), 2
            ),
            "avg_transaction_count_retained": round(
                float(retained_df["Total_Trans_Ct"].mean()), 2
            ),
            "avg_inactive_months_churned": round(
                float(churned_df["Months_Inactive_12_mon"].mean()), 2
            ),
            "avg_inactive_months_retained": round(
                float(retained_df["Months_Inactive_12_mon"].mean()), 2
            ),
            "avg_utilization_churned": round(
                float(churned_df["Avg_Utilization_Ratio"].mean()), 3
            ),
            "avg_utilization_retained": round(
                float(retained_df["Avg_Utilization_Ratio"].mean()), 3
            ),
        },
        "segment_risk_snapshot": {
            "highest_risk_card_category": highest_risk_card,
            "highest_risk_card_churn_rate_percent": highest_risk_card_rate,
        },
    }
