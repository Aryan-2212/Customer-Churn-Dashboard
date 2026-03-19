import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_APP_DIR = PROJECT_ROOT / "web_app"
if str(WEB_APP_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_APP_DIR))

from analytics_context import build_behavioral_summary
from prompt_template import build_dashboard_context_payload, build_system_prompt


class DashboardContextTests(unittest.TestCase):
    def test_behavioral_summary_contains_expected_keys(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "Attrition_Flag": 1,
                    "Card_Category": "Blue",
                    "Total_Trans_Amt": 1200,
                    "Total_Trans_Ct": 40,
                    "Months_Inactive_12_mon": 3,
                    "Avg_Utilization_Ratio": 0.4,
                },
                {
                    "Attrition_Flag": 0,
                    "Card_Category": "Silver",
                    "Total_Trans_Amt": 3400,
                    "Total_Trans_Ct": 68,
                    "Months_Inactive_12_mon": 1,
                    "Avg_Utilization_Ratio": 0.2,
                },
            ]
        )

        summary = build_behavioral_summary(df)

        self.assertIn("behavioral_patterns", summary)
        self.assertIn("segment_risk_snapshot", summary)
        self.assertIn("highest_risk_card_category", summary["segment_risk_snapshot"])

    def test_system_prompt_embeds_dashboard_context(self) -> None:
        payload = build_dashboard_context_payload(
            dataset_size=100,
            churn_rate=16.07,
            churned_customers=16,
            best_model="Random Forest",
            top_drivers=["Total_Trans_Amt", "Months_Inactive_12_mon"],
            dashboard_pages=[("Customer Overview", "Overall KPIs")],
        )

        prompt = build_system_prompt(payload)

        self.assertIn("customer churn analytics dashboard", prompt.lower())
        self.assertIn("Random Forest", prompt)
        self.assertIn("Customer Overview", prompt)


if __name__ == "__main__":
    unittest.main()
