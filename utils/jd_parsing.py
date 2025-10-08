# jd_parsing.py - Job description parsing constants and patterns
import re
from typing import Dict, Optional, Tuple

# ---------- Constant for app.py ----------
# Role keywords for job title detection
ROLE_KEYWORDS = [
    "artificial intelligence engineer", "ai engineer", "ml engineer", "machine learning engineer",
    "data scientist", "senior data scientist", "data engineer", "software engineer",
    "mlops engineer", "research scientist", "nlp engineer", "computer vision engineer",
    "generative ai engineer", "deep learning engineer"
]

# Regex patterns for extracting job titles
TITLE_REGEXES = [
    r"(?im)^\s*job\s*title\s*[:\-]\s*(.+)$",
    r"(?im)^\s*title\s*[:\-]\s*(.+)$",
    r"(?im)^\s*(?:position|role)\s*[:\-]\s*(.+)$",
]

# Noise line hints to filter out irrelevant content
NOISE_LINE_HINTS = [
    "logo", "share", "save", "easy apply", "promoted", "actively reviewing",
    "on-site", "onsite", "hybrid", "remote", "full-time", "part-time",
    "contract", "matches your job preferences", "message", "about the job",
    "rate-", "location-", "multiple locations", "apply", "premium", "job poster",
    "our benefits", "learn more", "company website", "company description", "company overview",
    "company values", "company culture", "company history", "company mission", "company vision",
]

CITY_REGEX = re.compile(r"^[A-Za-z][A-Za-z\s\-\.\']{1,48}$")


# ---------- Constant for provider.py ----------
_TRAIN_SECTORS = {
    "Arts, Entertainment & Recreation",
    "Construction, Repair & Maintenance",
    "Oil, Gas, Energy & Utilities",
    "Accounting & Legal",
    "Aerospace & Defense",
    "Agriculture & Forestry",
    "Biotech & Pharmaceuticals",
    "Business Services",
    "Consumer Services",
    "Education",
    "Finance",
    "Government",
    "Health Care",
    "Information Technology",
    "Insurance",
    "Manufacturing",
    "Media",
    "Mining & Metals",
    "Non-Profit",
    "Real Estate",
    "Retail",
    "Telecommunications",
    "Transportation & Logistics",
    "Travel & Tourism",
}

# TODO, need to be extended
_SECTOR_MAP = {
    # --- MEDIA vs ARTS ---
    # Companies producing/distributing content → Media
    "streaming":                         "Media",
    "film":                              "Media",
    "television":                        "Media",
    "tv":                                "Media",
    "music":                             "Media",
    "publishing":                        "Media",
    "radio":                             "Media",
    "news":                              "Media",
    "media":                             "Media",
    
    
    # Venues/experiences → Arts, Entertainment & Recreation
    "theme park":                        "Arts, Entertainment & Recreation",
    "amusement park":                    "Arts, Entertainment & Recreation",
    "museum":                            "Arts, Entertainment & Recreation",
    "performing arts":                   "Arts, Entertainment & Recreation",
    "recreation":                        "Arts, Entertainment & Recreation",
    "sports club":                       "Arts, Entertainment & Recreation",

    # --- INFORMATION TECHNOLOGY ---
    "software":                          "Information Technology",
    "application software":              "Information Technology",
    "computer software":                 "Information Technology",
    "information technology":            "Information Technology",
    "it services":                       "Information Technology",
    "cloud":                             "Information Technology",
    "cloud computing":                   "Information Technology",
    "saas":                              "Information Technology",
    "platform":                          "Information Technology",
    "data analytics":                    "Information Technology",
    "data warehousing":                  "Information Technology",
    "data cloud":                        "Information Technology",
    "internet":                          "Information Technology",
    "social media":                      "Information Technology",
    "ai":                                "Information Technology",
    "artificial intelligence":           "Information Technology",
    "machine learning":                  "Information Technology",
    "big data":                          "Information Technology",
    "cybersecurity":                     "Information Technology",
    "semiconductor":                     "Information Technology",
    "semiconductors":                    "Information Technology",
    "computer hardware":                 "Information Technology",

    # --- FINANCE (incl. fintech/payments) ---
    "financial technology":              "Finance",
    "fintech":                           "Finance",
    "payments":                          "Finance",
    "payment processing":                "Finance",
    "credit card":                       "Finance",
    "bank":                              "Finance",
    "banking":                           "Finance",
    "asset management":                  "Finance",
    "investment management":             "Finance",
    "brokerage":                         "Finance",
    "lending":                           "Finance",

    # --- CONSUMER SERVICES vs RETAIL ---
    # Delivery/food/ride-hailing/restaurants → Consumer Services
    "food delivery":                     "Consumer Services",
    "delivery":                          "Consumer Services",
    "on-demand delivery":                "Consumer Services",
    "grocery delivery":                  "Consumer Services",
    "ride-hailing":                      "Consumer Services",
    "ride sharing":                      "Consumer Services",
    "restaurant":                        "Consumer Services",
    "hospitality services":              "Consumer Services",
    # Retail/e-commerce → Retail
    "e-commerce":                        "Retail",
    "ecommerce":                         "Retail",
    "retail":                            "Retail",
    "online retail":                     "Retail",

    # --- TRANSPORTATION & LOGISTICS vs TRAVEL & TOURISM ---
    # Logistics/freight/shipping/airlines/rail → Transportation & Logistics
    "logistics":                         "Transportation & Logistics",
    "freight":                           "Transportation & Logistics",
    "shipping":                          "Transportation & Logistics",
    "airline":                           "Transportation & Logistics",
    "rail":                              "Transportation & Logistics",
    "trucking":                          "Transportation & Logistics",
    "mobility":                          "Transportation & Logistics",
    "transportation":                    "Transportation & Logistics",
    "supply chain":                      "Transportation & Logistics",
    # Hotels/resorts/travel booking/tourism → Travel & Tourism
    "hotel":                             "Travel & Tourism",
    "resort":                            "Travel & Tourism",
    "hospitality":                       "Travel & Tourism",
    "tourism":                           "Travel & Tourism",
    "travel agency":                     "Travel & Tourism",
    "online travel":                     "Travel & Tourism",
    "booking":                           "Travel & Tourism",

    # --- ENERGY / UTILITIES ---
    "energy":                            "Oil, Gas, Energy & Utilities",
    "oil":                               "Oil, Gas, Energy & Utilities",
    "gas":                               "Oil, Gas, Energy & Utilities",
    "utilities":                         "Oil, Gas, Energy & Utilities",
    "power":                             "Oil, Gas, Energy & Utilities",
    "renewable":                         "Oil, Gas, Energy & Utilities",

    # --- HEALTH & BIO ---
    "health care":                       "Health Care",
    "healthcare":                        "Health Care",
    "hospital":                          "Health Care",
    "medical device":                    "Health Care",
    "biotech":                           "Biotech & Pharmaceuticals",
    "biotechnology":                     "Biotech & Pharmaceuticals",
    "pharmaceutical":                    "Biotech & Pharmaceuticals",
    "pharmaceuticals":                   "Biotech & Pharmaceuticals",

    # --- TELECOM ---
    "telecommunications":                "Telecommunications",
    "telecom":                           "Telecommunications",
    "wireless":                          "Telecommunications",
    "mobile network":                    "Telecommunications",
    "isp":                               "Telecommunications",
    "broadband":                         "Telecommunications",

    # --- REAL ESTATE ---
    "real estate":                       "Real Estate",
    "property":                          "Real Estate",
    "reit":                              "Real Estate",
    "reit (real estate investment trust)": "Real Estate",
    "real estate development":            "Real Estate",
    "real estate investment":             "Real Estate",
    "property management":                "Real Estate",
    "property development":               "Real Estate",
    "commercial real estate":             "Real Estate",
    "residential real estate":            "Real Estate",
    "industrial real estate":             "Real Estate",
    "real estate services":               "Real Estate",
    "realty":                             "Real Estate",
    "brokerage":                          "Real Estate",
    "real estate brokerage":              "Real Estate",
    "real estate agency":                 "Real Estate",
    "housing":                            "Real Estate",
    "mortgage":                           "Real Estate",
    "leasing":                            "Real Estate",
    "land development":                   "Real Estate",
    "urban development":                  "Real Estate",
    "construction & real estate":         "Real Estate",
    "property consulting":                "Real Estate",
    "real estate consulting":             "Real Estate",
    "facilities management":              "Real Estate",
    "property leasing":                   "Real Estate",
    "real estate management":             "Real Estate",
    "real estate investment trust":       "Real Estate",

    # --- INSURANCE ---
    "insurance":                         "Insurance",
    
    # --- MANUFACTURING ---
    "manufacturing":                     "Manufacturing",
    "factory":                           "Manufacturing",
    "factories":                         "Manufacturing",
    "production":                        "Manufacturing",
    "industrial":                        "Manufacturing",
    "industrial manufacturing":          "Manufacturing",
    "industrial machinery":              "Manufacturing",
    "machinery":                         "Manufacturing",
    "automotive manufacturing":          "Manufacturing",
    "automobile":                        "Manufacturing",
    "auto parts":                        "Manufacturing",
    "automotive":                        "Manufacturing",
    "electronics manufacturing":         "Manufacturing",
    "semiconductor manufacturing":       "Manufacturing",
    "semiconductors":                    "Manufacturing",
    "hardware manufacturing":            "Manufacturing",
    "equipment manufacturing":           "Manufacturing",
    "chemical manufacturing":            "Manufacturing",
    "pharmaceutical manufacturing":      "Manufacturing",
    "food manufacturing":                "Manufacturing",
    "food production":                   "Manufacturing",
    "consumer goods":                    "Manufacturing",
    "textiles":                          "Manufacturing",
    "apparel":                           "Manufacturing",
    "furniture":                         "Manufacturing",
    "plastics":                          "Manufacturing",
    "packaging":                         "Manufacturing",
    "printing":                          "Manufacturing",
    "fabrication":                       "Manufacturing",
    "3d printing":                       "Manufacturing",
    "lean manufacturing":                "Manufacturing",
    "assembly":                          "Manufacturing",
    "supply chain manufacturing":        "Manufacturing",
    
    # --- MINING & METALS ---
    "mining":                            "Mining & Metals",
    "metals":                            "Mining & Metals",
    "metal":                             "Mining & Metals",
    "steel":                             "Mining & Metals",
    "iron":                              "Mining & Metals",
    "aluminum":                          "Mining & Metals",
    "copper":                            "Mining & Metals",
    "gold":                              "Mining & Metals",
    "silver":                            "Mining & Metals",
    "coal":                              "Mining & Metals",
    "ore":                               "Mining & Metals",
    "extraction":                        "Mining & Metals",
    "mining & metals":                   "Mining & Metals",
    "mining/metal":                      "Mining & Metals",
    "natural resources":                 "Mining & Metals",
    "raw materials":                     "Mining & Metals",
    "refining":                          "Mining & Metals",
    "smelting":                          "Mining & Metals",
    "drilling":                          "Mining & Metals",
    "oil drilling":                      "Mining & Metals",
    "metal fabrication":                 "Mining & Metals",
    "materials engineering":             "Mining & Metals",
    "metallurgy":                        "Mining & Metals",


    # --- BUSINESS SERVICES ---
    "business services":                 "Business Services",
    "bpo":                               "Business Services",
    "outsourcing":                       "Business Services",
    "consulting":                        "Business Services",
    "management consulting":             "Business Services",
    "strategy consulting":               "Business Services",
    "it consulting":                     "Business Services",
    "professional services":             "Business Services",
    "corporate services":                "Business Services",
    "staffing":                          "Business Services",
    "recruiting":                        "Business Services",
    "human resources":                   "Business Services",
    "talent acquisition":                "Business Services",
    "hr consulting":                     "Business Services",
    "facilities services":               "Business Services",
    "business process outsourcing":      "Business Services",
    "administrative services":           "Business Services",
    "back office":                       "Business Services",
    "shared services":                   "Business Services",
    "customer support":                  "Business Services",
    "call center":                       "Business Services",
    "virtual assistant":                 "Business Services",
    "business consulting":               "Business Services",
    "operations consulting":             "Business Services",
    "advisory services":                 "Business Services",
    "corporate training":                "Business Services",
    "business management":               "Business Services",
    
    # --- CONSTRUCTION & REPAIR & MAINTENANCE ---
    "construction":                      "Construction, Repair & Maintenance",
    "general contracting":               "Construction, Repair & Maintenance",
    "contracting":                       "Construction, Repair & Maintenance",
    "building":                          "Construction, Repair & Maintenance",
    "engineering & construction":        "Construction, Repair & Maintenance",
    "civil engineering":                 "Construction, Repair & Maintenance",
    "architectural services":            "Construction, Repair & Maintenance",
    "architecture":                      "Construction, Repair & Maintenance",
    "interior design":                   "Construction, Repair & Maintenance",
    "remodeling":                        "Construction, Repair & Maintenance",
    "renovation":                        "Construction, Repair & Maintenance",
    "repair":                            "Construction, Repair & Maintenance",
    "maintenance":                       "Construction, Repair & Maintenance",
    "property maintenance":              "Construction, Repair & Maintenance",
    "facility maintenance":              "Construction, Repair & Maintenance",
    "hvac":                              "Construction, Repair & Maintenance",
    "plumbing":                          "Construction, Repair & Maintenance",
    "electrical contracting":            "Construction, Repair & Maintenance",
    "mechanical contracting":            "Construction, Repair & Maintenance",
    "roofing":                           "Construction, Repair & Maintenance",
    "carpentry":                         "Construction, Repair & Maintenance",
    "painting":                          "Construction, Repair & Maintenance",
    "real estate development":           "Construction, Repair & Maintenance",
    "construction management":           "Construction, Repair & Maintenance",
    
    # --- ACCOUNTING & LEGAL ---
    "accounting":                        "Accounting & Legal",
    "audit":                             "Accounting & Legal",
    "auditing":                          "Accounting & Legal",
    "bookkeeping":                       "Accounting & Legal",
    "tax":                               "Accounting & Legal",
    "tax services":                      "Accounting & Legal",
    "tax consulting":                    "Accounting & Legal",
    "financial audit":                   "Accounting & Legal",
    "public accounting":                 "Accounting & Legal",
    "cpa":                               "Accounting & Legal",
    "chartered accountant":              "Accounting & Legal",
    "legal":                             "Accounting & Legal",
    "law":                               "Accounting & Legal",
    "law firm":                          "Accounting & Legal",
    "attorney":                          "Accounting & Legal",
    "law practice":                      "Accounting & Legal",
    "corporate law":                     "Accounting & Legal",
    "intellectual property":             "Accounting & Legal",
    "litigation":                        "Accounting & Legal",
    "paralegal":                         "Accounting & Legal",
    "notary":                            "Accounting & Legal",
    "legal services":                    "Accounting & Legal",
    "compliance":                        "Accounting & Legal",
    "regulatory":                        "Accounting & Legal",
    "risk & compliance":                 "Accounting & Legal",
    "forensic accounting":               "Accounting & Legal",

    # --- GOVERNMENT ---
    "government":                        "Government",
    "public administration":             "Government",
    "federal government":                "Government",
    "state government":                  "Government",
    "local government":                  "Government",
    "city government":                   "Government",
    "county government":                 "Government",
    "national government":               "Government",
    "civil service":                     "Government",
    "public sector":                     "Government",
    "municipal":                         "Government",
    "defense":                           "Government",
    "armed forces":                      "Government",
    "military":                          "Government",
    "public safety":                     "Government",
    "law enforcement":                   "Government",
    
    # --- NON-PROFIT ---
    "non-profit":                        "Non-Profit",
    "nonprofit":                         "Non-Profit",
    "not for profit":                    "Non-Profit",
    "not-for-profit":                    "Non-Profit",
    "ngo":                               "Non-Profit",
    "charity":                           "Non-Profit",
    "foundation":                        "Non-Profit",
    "humanitarian organization":         "Non-Profit",
    "social services":                   "Non-Profit",
    "educational nonprofit":             "Non-Profit",
    "environmental nonprofit":           "Non-Profit",
    "community organization":            "Non-Profit",
    "philanthropy":                      "Non-Profit",
    "voluntary organization":            "Non-Profit",
    "religious organization":            "Non-Profit",
    "church":                            "Non-Profit",
    "faith-based":                       "Non-Profit",

    # --- AEROSPACE ---
    "aerospace":                         "Aerospace & Defense",
    "defense":                           "Aerospace & Defense",
    "defence":                           "Aerospace & Defense",
    "aviation":                          "Aerospace & Defense",
    "airlines":                          "Aerospace & Defense",
    "airline":                           "Aerospace & Defense",
    "space":                             "Aerospace & Defense",
    "space technology":                  "Aerospace & Defense",
    "aerospace engineering":             "Aerospace & Defense",
    "military":                          "Aerospace & Defense",
    "national security":                 "Aerospace & Defense",
    "aerospace & defense":               "Aerospace & Defense",
    "aerospace/defense":                 "Aerospace & Defense",
    "aircraft manufacturing":            "Aerospace & Defense",
    "missile systems":                   "Aerospace & Defense",
    "satellite":                         "Aerospace & Defense",
    "drone":                             "Aerospace & Defense",
    "uav":                               "Aerospace & Defense",
    "aerospace technology":              "Aerospace & Defense",
    
    # --- AGRICULTURE ---
    "agriculture":                       "Agriculture & Forestry",
    "forestry":                          "Agriculture & Forestry",
    "farming":                           "Agriculture & Forestry",
    "agtech":                            "Agriculture & Forestry",
    "agribusiness":                      "Agriculture & Forestry",
    "crop production":                   "Agriculture & Forestry",
    "livestock":                         "Agriculture & Forestry",
    "fishery":                           "Agriculture & Forestry",
    "fisheries":                         "Agriculture & Forestry",
    "ranching":                          "Agriculture & Forestry",
    "horticulture":                      "Agriculture & Forestry",
    "aquaculture":                       "Agriculture & Forestry",
    "sustainable agriculture":            "Agriculture & Forestry",
    "precision agriculture":              "Agriculture & Forestry",
    "agricultural engineering":           "Agriculture & Forestry",
    "agricultural technology":            "Agriculture & Forestry",
    "agriculture & forestry":             "Agriculture & Forestry",
    "agriculture/forestry":               "Agriculture & Forestry",
}


# ---------- helper functions ----------
# City name validation regex
def parse_years_experience(text: str) -> Optional[int]:
    # Matches "3 years", "5+ years", "7 yrs"
    m = re.search(r'(\d+)\s*\+?\s*(?:years?|yrs?)', text, flags=re.I)
    if m:
        return int(m.group(1))
    return None

def parse_education(text: str) -> Optional[str]:
    levels = ["Bachelor", "Master", "PhD", "Doctorate", "MBA", "Associate", "High School"]
    for lvl in levels:
        if re.search(lvl, text, re.I):
            return lvl
    return None

def parse_location(text: str) -> Optional[str]:
    # Very naive: looks for "City, ST" pattern
    m = re.search(r'([A-Z][a-z]+(?: [A-Z][a-z]+)*,\s*[A-Z]{2})', text)
    if m:
        return m.group(1)
    return None

def parse_salary_range(text: str) -> Optional[Tuple[int,int]]:
    """
    Extract "$80,000 - $120,000" or "$50K–90K" style ranges.
    Returns (low, high) annual USD.
    """
    text = text.replace(",", "")
    # $80,000 - $120,000
    m = re.search(r'\$?(\d{2,6})(?:\s*-\s*|\s*–\s*)\$?(\d{2,6})', text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        # normalize: if small numbers, assume K
        if lo < 1000: lo *= 1000
        if hi < 1000: hi *= 1000
        return lo, hi
    # $50K-90K
    m = re.search(r'\$?(\d+)\s*[kK]\s*[-–]\s*\$?(\d+)\s*[kK]', text)
    if m:
        return int(m.group(1))*1000, int(m.group(2))*1000
    return None

def parse_jd(jd: str) -> Dict[str, Optional[str]]:
    return {
        "years_experience": parse_years_experience(jd),
        "education_level": parse_education(jd),
        "location": parse_location(jd),
        "salary_range": parse_salary_range(jd),
    }
