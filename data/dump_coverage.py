"""FineWeb (Common Crawl) dump -> covered calendar months, and per-month token budgets.
Used by get_train_set.py and verify_8B_tokens.py; running it writes tokens_dist.csv to the cwd.

Usage: python data/dump_coverage.py
"""
# Months missing here are skipped; the next month's multiplier absorbs their budget.
# Inline comments give Common Crawl's own name for each crawl.
COVERAGE = {
            "CC-MAIN-2013-20": ["2013-05", "2013-06"],        # Summer 2013
            "CC-MAIN-2013-48": ["2013-12"],        # Winter 2013
            "CC-MAIN-2014-10": ["2014-03"],                   # March 2014
            "CC-MAIN-2014-15": ["2014-04"],                   # April 2014
            "CC-MAIN-2014-23": ["2014-07"],                   # July 2014
            "CC-MAIN-2014-35": ["2014-08"],                   # August 2014
            "CC-MAIN-2014-41": ["2014-09"],                   # September 2014
            "CC-MAIN-2014-42": ["2014-10"],                   # October 2014
            "CC-MAIN-2014-49": ["2014-11"],                   # November 2014
            "CC-MAIN-2014-52": ["2014-12"],                   # December 2014
            "CC-MAIN-2015-06": ["2015-01"],                   # January 2015
            "CC-MAIN-2015-11": ["2015-02"],                   # February 2015
            "CC-MAIN-2015-14": ["2015-03"],                   # March 2015
            "CC-MAIN-2015-18": ["2015-04"],                   # April 2015
            "CC-MAIN-2015-22": ["2015-05"],                   # May 2015
            "CC-MAIN-2015-27": ["2015-06"],                   # June 2015
            "CC-MAIN-2015-32": ["2015-07"],                   # July 2015
            "CC-MAIN-2015-35": ["2015-08"],                   # August 2015
            "CC-MAIN-2015-40": ["2015-10"],                   # September 2015
            "CC-MAIN-2015-48": ["2015-11"],                   # November 2015
            "CC-MAIN-2016-07": ["2016-02"],                   # February 2016
            "CC-MAIN-2016-18": ["2016-04"],                   # April 2016
            "CC-MAIN-2016-22": ["2016-05"],                   # May 2016
            "CC-MAIN-2016-26": ["2016-06"],                   # June 2016
            "CC-MAIN-2016-30": ["2016-07"],                   # July 2016
            "CC-MAIN-2016-36": ["2016-08"],                   # August 2016
            "CC-MAIN-2016-40": ["2016-09"],                   # September 2016
            "CC-MAIN-2016-44": ["2016-10"],                   # October 2016
            "CC-MAIN-2016-50": ["2016-12"],                   # December 2016
            "CC-MAIN-2017-04": ["2017-01"],                   # January 2017
            "CC-MAIN-2017-09": ["2017-02"],                   # February 2017
            "CC-MAIN-2017-13": ["2017-03"],                   # March 2017
            "CC-MAIN-2017-17": ["2017-04"],                   # April 2017
            "CC-MAIN-2017-22": ["2017-05"],                   # May 2017
            "CC-MAIN-2017-26": ["2017-06"],                   # June 2017
            "CC-MAIN-2017-30": ["2017-07"],                   # July 2017
            "CC-MAIN-2017-34": ["2017-08"],                   # August 2017
            "CC-MAIN-2017-39": ["2017-09"],                   # September 2017
            "CC-MAIN-2017-43": ["2017-10"],                   # October 2017
            "CC-MAIN-2017-47": ["2017-11"],                   # November 2017
            "CC-MAIN-2017-51": ["2017-12"],                   # December 2017
            "CC-MAIN-2018-05": ["2018-01"],                   # January 2018
            "CC-MAIN-2018-09": ["2018-02"],                   # February 2018
            "CC-MAIN-2018-13": ["2018-03"],                   # March 2018
            "CC-MAIN-2018-17": ["2018-04"],                   # April 2018
            "CC-MAIN-2018-22": ["2018-05"],                   # May 2018
            "CC-MAIN-2018-26": ["2018-06"],                   # June 2018
            "CC-MAIN-2018-30": ["2018-07"],                   # July 2018
            "CC-MAIN-2018-34": ["2018-08"],                   # August 2018
            "CC-MAIN-2018-39": ["2018-09"],                   # September 2018
            "CC-MAIN-2018-43": ["2018-10"],                   # October 2018
            "CC-MAIN-2018-47": ["2018-11"],                   # November 2018
            "CC-MAIN-2018-51": ["2018-12"],                   # December 2018
            "CC-MAIN-2019-04": ["2019-01"],                   # January 2019
            "CC-MAIN-2019-09": ["2019-02"],                   # February 2019
            "CC-MAIN-2019-13": ["2019-03"],                   # March 2019
            "CC-MAIN-2019-18": ["2019-04"],                   # April 2019
            "CC-MAIN-2019-22": ["2019-05"],                   # May 2019
            "CC-MAIN-2019-26": ["2019-06"],                   # June 2019
            "CC-MAIN-2019-30": ["2019-07"],                   # July 2019
            "CC-MAIN-2019-35": ["2019-08"],                   # August 2019
            "CC-MAIN-2019-39": ["2019-09"],                   # September 2019
            "CC-MAIN-2019-43": ["2019-10"],                   # October 2019
            "CC-MAIN-2019-47": ["2019-11"],                   # November 2019
            "CC-MAIN-2019-51": ["2019-12"],                   # December 2019
            "CC-MAIN-2020-05": ["2020-01"],                   # January 2020
            "CC-MAIN-2020-10": ["2020-02"],                   # February 2020
            "CC-MAIN-2020-16": ["2020-03", "2020-04"],        # March/April 2020
            "CC-MAIN-2020-24": ["2020-05", "2020-06"],        # May/June 2020
            "CC-MAIN-2020-29": ["2020-07"],                   # July 2020
            "CC-MAIN-2020-34": ["2020-08"],                   # August 2020
            "CC-MAIN-2020-40": ["2020-09"],                   # September 2020
            "CC-MAIN-2020-45": ["2020-10"],                   # October 2020
            "CC-MAIN-2020-50": ["2020-11", "2020-12"],        # Nov/Dec 2020
            "CC-MAIN-2021-04": ["2021-01"],                   # January 2021
            "CC-MAIN-2021-10": ["2021-02", "2021-03"],        # Feb/Mar 2021
            "CC-MAIN-2021-17": ["2021-04"],                   # April 2021
            "CC-MAIN-2021-21": ["2021-05"],                   # May 2021
            "CC-MAIN-2021-25": ["2021-06"],                   # June 2021
            "CC-MAIN-2021-31": ["2021-07", "2021-08"],        # July/Aug 2021
            "CC-MAIN-2021-39": ["2021-09"],                   # September 2021
            "CC-MAIN-2021-43": ["2021-10"],                   # October 2021
            "CC-MAIN-2021-49": ["2021-11", "2021-12"],        # Nov/Dec 2021
            "CC-MAIN-2022-05": ["2022-01"],                   # January 2022
            "CC-MAIN-2022-21": ["2022-05"],                   # May 2022
            "CC-MAIN-2022-27": ["2022-06", "2022-07"],        # June/July 2022
            "CC-MAIN-2022-33": ["2022-08"],                   # August 2022
            "CC-MAIN-2022-40": ["2022-09", "2022-10"],        # Sept/Oct 2022
            "CC-MAIN-2022-49": ["2022-11", "2022-12"],        # Nov/Dec 2022
            "CC-MAIN-2023-06": ["2023-01", "2023-02"],        # Jan/Feb 2023
            "CC-MAIN-2023-14": ["2023-03", "2023-04"],        # Mar/Apr 2023
            "CC-MAIN-2023-23": ["2023-05", "2023-06"],        # May/June 2023
            "CC-MAIN-2023-40": ["2023-09", "2023-10"],        # Sept/Oct 2023
            "CC-MAIN-2023-50": ["2023-11", "2023-12"],        # Nov/Dec 2023
            "CC-MAIN-2024-10": ["2024-02", "2024-03"],        # Feb/Mar 2024
            "CC-MAIN-2024-18": ["2024-04"],                   # April 2024
            "CC-MAIN-2024-22": ["2024-05"],                   # May 2024
            "CC-MAIN-2024-26": ["2024-06"],                   # June 2024
            "CC-MAIN-2024-30": ["2024-07"],                   # July 2024
            "CC-MAIN-2024-33": ["2024-08"],                   # August 2024
            "CC-MAIN-2024-38": ["2024-09"],                   # September 2024
            "CC-MAIN-2024-42": ["2024-10"],                   # October 2024
            "CC-MAIN-2024-46": ["2024-11"],                   # November 2024
            "CC-MAIN-2024-51": ["2024-12"],                   # December 2024
            "CC-MAIN-2025-05": ["2025-01"],                   # January 2025
            "CC-MAIN-2025-08": ["2025-02"],                   # February 2025
            "CC-MAIN-2025-13": ["2025-03"],                   # March 2025
            "CC-MAIN-2025-18": ["2025-04"],                   # April 2025
            "CC-MAIN-2025-21": ["2025-05"],                   # May 2025
            "CC-MAIN-2025-26": ["2025-06"]                    # June 2025
        }

# Precomputed month -> {config, multiplier}, so each month can be built independently.
def _build_month_multiplier():
    """Build a dict mapping month -> (config, multiplier) for parallel deployment."""
    # Invert COVERAGE into month -> dump name
    month_to_config = {}
    for config, months in COVERAGE.items():
        for month in months:
            month_to_config[month] = config
    
    # Get sorted months for gap calculation
    sorted_months = sorted(month_to_config.keys())
    
    # Multiplier = 1 + number of uncovered months since the previous covered month
    result = {}
    prev_month = None
    for year_month in sorted_months:
        if prev_month is None:
            multiplier = 1
        else:
            y1, m1 = map(int, prev_month.split("-"))
            y2, m2 = map(int, year_month.split("-"))
            diff = (y2 - y1) * 12 + (m2 - m1)
            gap = max(diff - 1, 0)
            multiplier = gap + 1
        
        result[year_month] = {
            "config": month_to_config[year_month],
            "multiplier": multiplier
        }
        prev_month = year_month
    
    return result

MONTH_MULTIPLIER = _build_month_multiplier()

def month_diff(m1: str, m2: str) -> int:
    """Compute the number of months between two YYYY-MM strings."""
    y1, mo1 = map(int, m1.split("-"))
    y2, mo2 = map(int, m2.split("-"))
    return (y2 - y1) * 12 + (mo2 - mo1)

def compute_cumulative_tokens(tokens_per_month: int = 1_150_000_000):
    """
    Compute cumulative tokens for each month using the gap-based multiplier.
    
    Returns a list of (date, tokens_this_month, cumulative_tokens) tuples.
    """
    # Extract all unique months in chronological order
    all_months = set()
    for months in COVERAGE.values():
        all_months.update(months)
    sorted_months = sorted(all_months)
    
    results = []
    cumulative = 0
    prev_month = None
    
    for year_month in sorted_months:
        if prev_month is None:
            multiplier = 1
        else:
            diff = month_diff(prev_month, year_month)
            gap = max(diff - 1, 0)
            multiplier = gap + 1
        
        tokens_this_month = tokens_per_month * multiplier
        cumulative += tokens_this_month
        results.append((year_month, tokens_this_month, cumulative))
        prev_month = year_month
    
    return results

def save_tokens_dist_csv(output_path: str = "tokens_dist.csv", tokens_per_month: int = 1_150_000_000):
    """Save cumulative token distribution to CSV."""
    import csv
    
    results = compute_cumulative_tokens(tokens_per_month)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "tokens_month", "cumulative_tokens"])
        for date, tokens_month, cumulative in results:
            writer.writerow([date, tokens_month, cumulative])
    
    print(f"✅ Wrote token distribution to {output_path}")

if __name__ == "__main__":
    save_tokens_dist_csv()
