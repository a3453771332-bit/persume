"""Business labels for fields already present in the frozen Home Credit data."""
FIELDS = {
    "DAYS_BIRTH": {"business_name": "Age", "category": "Personal", "description": "Age at application"},
    "DAYS_EMPLOYED": {"business_name": "Employment duration", "category": "Employment", "description": "Recorded employment duration"},
    "AMT_INCOME_TOTAL": {"business_name": "Monthly income", "category": "Financial", "description": "Annual income divided by 12"},
    "AMT_CREDIT": {"business_name": "Requested loan amount", "category": "Financial", "description": "Requested credit amount"},
}
