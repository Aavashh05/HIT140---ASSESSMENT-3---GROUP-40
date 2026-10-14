from pathlib import Path
import re
import unicodedata

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_WINDOW = 4


TEAM_SLUG_OVERRIDES = {
    "Bosnia–Herz": "Bosnia-and-Herzegovina",
    "Curaçao": "Curacao",
    "Côte d'Ivoire": "Cote-dIvoire",
    "IR Iran": "IR-Iran",
    "Türkiye": "Turkiye",
    "USA": "United-States",
}

RANKING_NAME_MAP = {
    "Bosnia–Herz": "Bosnia and Herzegovina",
    "Dominican Rep.": "Dominican Republic",
    "Gambia": "The Gambia",
    "Korea DPR": "DPR Korea",
    "No. Macedonia": "North Macedonia",
    "Rep. of Ireland": "Republic of Ireland",
    "Trin & Tobago": "Trinidad and Tobago",
    "UAE": "United Arab Emirates",
}


def team_slug(team):
    if team in TEAM_SLUG_OVERRIDES:
        return TEAM_SLUG_OVERRIDES[team]

    value = unicodedata.normalize("NFKD", team)
    value = value.encode("ascii", "ignore").decode()
    value = value.replace("–", "-").replace("'", "")
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")
    return value


def build_matches():
    matches = pd.read_csv(
        RAW_DIR / "SCORES & FIXTURES.csv",
        skiprows=4,
    )

    matches = matches.dropna(
        subset=["Round", "Date", "Home", "Away", "Score"]
    ).copy()

    matches["Team_A"] = (
        matches["Home"]
        .str.strip()
        .str.replace(r"\s+[a-z]{2,3}$", "", regex=True)
    )

    matches["Team_B"] = (
        matches["Away"]
        .str.strip()
        .str.replace(r"^[a-z]{2,3}\s+", "", regex=True)
    )

    scores = matches["Score"].str.extract(
        r"(\d+)\s*[–-]\s*(\d+)"
    )

    if scores.isna().any().any():
        raise ValueError("A match score could not be read.")

    matches["Goals_A"] = scores[0].astype(int)
    matches["Goals_B"] = scores[1].astype(int)
    matches["Goal_Difference"] = (
        matches["Goals_A"] - matches["Goals_B"]
    )

    matches.insert(0, "Match_ID", range(1, len(matches) + 1))

    matches = matches[
        [
            "Match_ID",
            "Date",
            "Round",
            "Team_A",
            "Team_B",
            "Goals_A",
            "Goals_B",
            "Goal_Difference",
        ]
    ]

    if len(matches) != 104:
        raise ValueError(f"Expected 104 matches, found {len(matches)}.")

    matches.to_csv(
        PROCESSED_DIR / "matches_master_104.csv",
        index=False,
    )

    return matches


def filter_tournament(data):
    rounds = {
        "Group stage",
        "Round of 32",
        "Round of 16",
        "Quarter-finals",
        "Semi-finals",
        "Third-place match",
        "Final",
    }

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])

    return data[
        data["round"].isin(rounds)
        & data["date"].between("2026-06-11", "2026-07-19")
    ].copy()


def build_team_match_stats(matches):
    matches = matches.copy()
    matches["Date"] = pd.to_datetime(matches["Date"])

    team_a = matches[
        [
            "Match_ID",
            "Date",
            "Round",
            "Team_A",
            "Team_B",
            "Goals_A",
            "Goals_B",
        ]
    ].copy()

    team_a.columns = [
        "Match_ID",
        "Date",
        "Round",
        "Team",
        "Opponent",
        "Goals_Scored",
        "Goals_Conceded",
    ]

    team_b = matches[
        [
            "Match_ID",
            "Date",
            "Round",
            "Team_B",
            "Team_A",
            "Goals_B",
            "Goals_A",
        ]
    ].copy()

    team_b.columns = team_a.columns

    team_matches = pd.concat(
        [team_a, team_b],
        ignore_index=True,
    )

    schedule = filter_tournament(
        pd.read_csv(RAW_DIR / "fbref_team_schedule_2026.csv")
    )
    shooting_for = filter_tournament(
        pd.read_csv(RAW_DIR / "fbref_shooting_for_2026.csv")
    )
    shooting_against = filter_tournament(
        pd.read_csv(RAW_DIR / "fbref_shooting_against_2026.csv")
    )

    for data in (schedule, shooting_for, shooting_against):
        if len(data) != 208:
            raise ValueError("Expected 208 tournament team rows.")
        if data.duplicated(["date", "opponent"]).any():
            raise ValueError("Duplicate date-opponent rows were found.")

    schedule = schedule[
        ["date", "opponent", "Poss", "venue", "result", "Notes", "match_report"]
    ].rename(
        columns={
            "date": "Date",
            "opponent": "Opponent",
            "Poss": "Possession",
            "venue": "Venue",
            "result": "Result",
        }
    )

    shooting_for = shooting_for[
        ["date", "opponent", "Standard_Sh", "Standard_SoT"]
    ].rename(
        columns={
            "date": "Date",
            "opponent": "Opponent",
            "Standard_Sh": "Shots_For",
            "Standard_SoT": "SoT_For",
        }
    )

    shooting_against = shooting_against[
        ["date", "opponent", "Standard_Sh", "Standard_SoT"]
    ].rename(
        columns={
            "date": "Date",
            "opponent": "Opponent",
            "Standard_Sh": "Shots_Against",
            "Standard_SoT": "SoT_Against",
        }
    )

    data = team_matches.merge(
        schedule,
        on=["Date", "Opponent"],
        how="left",
        validate="one_to_one",
    )

    data = data.merge(
        shooting_for,
        on=["Date", "Opponent"],
        how="left",
        validate="one_to_one",
    )

    data = data.merge(
        shooting_against,
        on=["Date", "Opponent"],
        how="left",
        validate="one_to_one",
    )

    required = [
        "Possession",
        "Shots_For",
        "SoT_For",
        "Shots_Against",
        "SoT_Against",
    ]

    if len(data) != 208 or not data.groupby("Match_ID").size().eq(2).all():
        raise ValueError("The team-match dataset must contain 208 rows.")

    if not data[required].notna().all().all():
        raise ValueError("Missing possession or shooting statistics.")

    for _, match in data.groupby("Match_ID"):
        first, second = match.iloc[0], match.iloc[1]
        if (
            first["Shots_Against"] != second["Shots_For"]
            or second["Shots_Against"] != first["Shots_For"]
            or first["SoT_Against"] != second["SoT_For"]
            or second["SoT_Against"] != first["SoT_For"]
        ):
            raise ValueError("Shooting statistics do not match between opponents.")

    data = data.sort_values(
        ["Date", "Match_ID", "Team"]
    ).reset_index(drop=True)

    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")

    data.to_csv(
        PROCESSED_DIR / "team_match_stats_208.csv",
        index=False,
    )

    return data


def prepare_rankings(matches):
    rankings = pd.read_csv(
        RAW_DIR / "fifa_rankings_11_june_2026.csv"
    )

    rankings["Team"] = rankings["Team"].replace(
        {"Bosnia and Herzegovina": "Bosnia–Herz"}
    )

    teams = sorted(set(matches["Team_A"]) | set(matches["Team_B"]))

    rankings = rankings[
        rankings["Team"].isin(teams)
    ].copy()

    rankings = rankings[
        [
            "Rank",
            "Team",
            "FIFA_Code",
            "FIFA_Points",
            "Previous_Rank",
            "Ranking_Movement",
            "Publication_Date",
            "Confederation",
        ]
    ].sort_values("Rank").reset_index(drop=True)

    if len(teams) != 48 or len(rankings) != 48:
        raise ValueError("Expected FIFA ranking data for all 48 teams.")

    if set(rankings["Team"]) != set(teams):
        raise ValueError("FIFA ranking team names do not match the tournament.")

    if rankings[["Rank", "FIFA_Points"]].isna().any().any():
        raise ValueError("Missing FIFA ranking values.")

    rankings.to_csv(
        PROCESSED_DIR / "fifa_rankings_world_cup_48.csv",
        index=False,
    )

    return rankings


def infer_tournament_team(row, team_slugs):
    report = str(row.get("match_report", ""))
    opponent = row.get("opponent")

    candidates = [
        team
        for team, slug in team_slugs.items()
        if slug in report
    ]

    remaining = [
        team for team in candidates if team != opponent
    ]

    if len(remaining) == 1:
        return remaining[0]

    return None


def parse_score_value(series):
    return pd.to_numeric(
        series.astype(str).str.extract(r"^(\d+)", expand=False),
        errors="coerce",
    )


def build_pre_tournament_history(matches, all_rankings):
    matches = matches.copy()
    matches["Date"] = pd.to_datetime(matches["Date"])

    teams = sorted(set(matches["Team_A"]) | set(matches["Team_B"]))
    team_slugs = {team: team_slug(team) for team in teams}

    opening_dates = pd.concat(
        [
            matches[["Date", "Team_A"]].rename(columns={"Team_A": "Team"}),
            matches[["Date", "Team_B"]].rename(columns={"Team_B": "Team"}),
        ],
        ignore_index=True,
    ).groupby("Team")["Date"].min()

    ranking_points = dict(
        zip(all_rankings["Team"], all_rankings["FIFA_Points"])
    )

    schedule = pd.read_csv(
        RAW_DIR / "fbref_team_schedule_2026.csv"
    )
    schedule["date"] = pd.to_datetime(schedule["date"])

    schedule["Team"] = schedule.apply(
        infer_tournament_team,
        axis=1,
        team_slugs=team_slugs,
    )

    schedule = schedule.dropna(subset=["Team"]).copy()

    schedule["Goals_Scored"] = parse_score_value(schedule["GF"])
    schedule["Goals_Conceded"] = parse_score_value(schedule["GA"])

    schedule = schedule[
        schedule.apply(
            lambda row: row["date"] < opening_dates[row["Team"]],
            axis=1,
        )
    ].copy()

    schedule = schedule.dropna(
        subset=["Goals_Scored", "Goals_Conceded", "result"]
    )

    schedule = schedule.sort_values(["Team", "date"])

    selected = (
        schedule.groupby("Team", group_keys=False)
        .tail(HISTORY_WINDOW)
        .copy()
    )

    counts = selected.groupby("Team").size()

    if len(counts) != 48 or not counts.eq(HISTORY_WINDOW).all():
        missing = {
            team: int(counts.get(team, 0))
            for team in teams
            if int(counts.get(team, 0)) != HISTORY_WINDOW
        }
        raise ValueError(
            f"Each team must have {HISTORY_WINDOW} pre-tournament matches: {missing}"
        )

    selected["Goal_Difference"] = (
        selected["Goals_Scored"] - selected["Goals_Conceded"]
    )
    selected["Win_Flag"] = (selected["result"] == "W").astype(int)
    selected["Draw_Flag"] = (selected["result"] == "D").astype(int)
    selected["Points"] = selected["result"].map({"W": 3, "D": 1, "L": 0})
    selected["Clean_Sheet"] = (selected["Goals_Conceded"] == 0).astype(int)
    selected["Failed_To_Score"] = (selected["Goals_Scored"] == 0).astype(int)

    selected["Opponent_Ranking_Name"] = selected["opponent"].replace(
        RANKING_NAME_MAP
    )
    selected["Opponent_FIFA_Points"] = selected["Opponent_Ranking_Name"].map(
        ranking_points
    )

    if selected["Opponent_FIFA_Points"].isna().any():
        missing = sorted(
            selected.loc[
                selected["Opponent_FIFA_Points"].isna(), "opponent"
            ].unique()
        )
        raise ValueError(
            f"Missing FIFA ranking points for pre-tournament opponents: {missing}"
        )

    selected["Strength_Adjusted_GD"] = (
        selected["Goal_Difference"]
        * selected["Opponent_FIFA_Points"]
        / 1500.0
    )

    selected = selected.rename(
        columns={
            "date": "Date",
            "opponent": "Opponent",
            "round": "Competition_Stage",
            "result": "Result",
        }
    )

    selected = selected[
        [
            "Team",
            "Date",
            "Opponent",
            "Competition_Stage",
            "Result",
            "Goals_Scored",
            "Goals_Conceded",
            "Goal_Difference",
            "Win_Flag",
            "Draw_Flag",
            "Points",
            "Clean_Sheet",
            "Failed_To_Score",
            "Opponent_FIFA_Points",
            "Strength_Adjusted_GD",
        ]
    ].sort_values(["Team", "Date"]).reset_index(drop=True)

    selected["Date"] = selected["Date"].dt.strftime("%Y-%m-%d")

    selected.to_csv(
        PROCESSED_DIR / "pre_tournament_recent_form_192.csv",
        index=False,
    )

    return selected


def build_prematch_features(team_matches, rankings, all_rankings, pre_tournament):
    tournament = team_matches.copy()
    tournament["Date"] = pd.to_datetime(tournament["Date"])

    tournament["Goal_Difference"] = (
        tournament["Goals_Scored"] - tournament["Goals_Conceded"]
    )
    tournament["Win_Flag"] = (
        tournament["Goals_Scored"] > tournament["Goals_Conceded"]
    ).astype(int)
    tournament["Draw_Flag"] = (
        tournament["Goals_Scored"] == tournament["Goals_Conceded"]
    ).astype(int)
    tournament["Points"] = (
        tournament["Win_Flag"] * 3 + tournament["Draw_Flag"]
    )
    tournament["Clean_Sheet"] = (
        tournament["Goals_Conceded"] == 0
    ).astype(int)
    tournament["Failed_To_Score"] = (
        tournament["Goals_Scored"] == 0
    ).astype(int)

    ranking_points = dict(
        zip(all_rankings["Team"], all_rankings["FIFA_Points"])
    )
    tournament["Opponent_Ranking_Name"] = tournament["Opponent"].replace(
        RANKING_NAME_MAP
    )
    tournament["Opponent_FIFA_Points_History"] = (
        tournament["Opponent_Ranking_Name"].map(ranking_points)
    )

    if tournament["Opponent_FIFA_Points_History"].isna().any():
        raise ValueError("Missing FIFA ranking points for tournament opponents.")

    tournament["Strength_Adjusted_GD"] = (
        tournament["Goal_Difference"]
        * tournament["Opponent_FIFA_Points_History"]
        / 1500.0
    )

    pre = pre_tournament.copy()
    pre["Date"] = pd.to_datetime(pre["Date"])

    history_columns = [
        "Team",
        "Date",
        "Goals_Scored",
        "Goals_Conceded",
        "Goal_Difference",
        "Win_Flag",
        "Draw_Flag",
        "Points",
        "Clean_Sheet",
        "Failed_To_Score",
        "Opponent_FIFA_Points",
        "Strength_Adjusted_GD",
    ]

    pre_history = pre[history_columns].copy()
    pre_history["Is_Tournament"] = 0
    pre_history["Match_ID"] = pd.NA

    tournament_history_source = tournament.copy().rename(
        columns={
            "Opponent_FIFA_Points_History": "Opponent_FIFA_Points"
        }
    )
    tournament_history = tournament_history_source[
        history_columns + ["Match_ID"]
    ].copy()
    tournament_history["Is_Tournament"] = 1

    combined = pd.concat(
        [pre_history, tournament_history],
        ignore_index=True,
    ).sort_values(["Team", "Date", "Is_Tournament"]).reset_index(drop=True)

    prior_rows = []

    for team, team_rows in combined.groupby("Team", sort=False):
        previous = []

        for _, row in team_rows.sort_values(["Date", "Is_Tournament"]).iterrows():
            if int(row["Is_Tournament"]) == 1:
                if len(previous) < HISTORY_WINDOW:
                    raise ValueError(
                        f"Insufficient pre-match history for {team} on {row['Date']}"
                    )

                window = pd.DataFrame(previous[-HISTORY_WINDOW:])

                prior_rows.append(
                    {
                        "Match_ID": int(row["Match_ID"]),
                        "Team": team,
                        "Prior_GF_Avg": window["Goals_Scored"].mean(),
                        "Prior_GA_Avg": window["Goals_Conceded"].mean(),
                        "Prior_GD_Avg": window["Goal_Difference"].mean(),
                        "Prior_Win_Rate": window["Win_Flag"].mean(),
                        "Prior_Draw_Rate": window["Draw_Flag"].mean(),
                        "Prior_Points_Per_Game": window["Points"].mean(),
                        "Prior_CleanSheet_Rate": window["Clean_Sheet"].mean(),
                        "Prior_FailedScore_Rate": window["Failed_To_Score"].mean(),
                        "Prior_Opp_FIFA_Points_Avg": window[
                            "Opponent_FIFA_Points"
                        ].mean(),
                        "Prior_Strength_Adjusted_GD_Avg": window[
                            "Strength_Adjusted_GD"
                        ].mean(),
                        "Rest_Days": (row["Date"] - window.iloc[-1]["Date"]).days,
                        "History_Window": HISTORY_WINDOW,
                    }
                )

            previous.append(row.to_dict())

    prior = pd.DataFrame(prior_rows)

    if len(prior) != 208:
        raise ValueError("Expected 208 pre-match history rows.")

    data = tournament.merge(
        prior,
        on=["Match_ID", "Team"],
        how="left",
        validate="one_to_one",
    )

    ranking_data = rankings[
        ["Team", "Rank", "FIFA_Points"]
    ].rename(columns={"Rank": "FIFA_Rank"})

    data = data.merge(
        ranking_data,
        on="Team",
        how="left",
        validate="many_to_one",
    )

    opponent_columns = [
        "Match_ID",
        "Team",
        "Prior_GF_Avg",
        "Prior_GA_Avg",
        "Prior_GD_Avg",
        "Prior_Win_Rate",
        "Prior_Draw_Rate",
        "Prior_Points_Per_Game",
        "Prior_CleanSheet_Rate",
        "Prior_FailedScore_Rate",
        "Prior_Opp_FIFA_Points_Avg",
        "Prior_Strength_Adjusted_GD_Avg",
        "Rest_Days",
        "FIFA_Rank",
        "FIFA_Points",
    ]

    opponent = data[opponent_columns].copy().rename(
        columns={
            "Team": "Opponent",
            "Prior_GF_Avg": "Opp_Prior_GF_Avg",
            "Prior_GA_Avg": "Opp_Prior_GA_Avg",
            "Prior_GD_Avg": "Opp_Prior_GD_Avg",
            "Prior_Win_Rate": "Opp_Prior_Win_Rate",
            "Prior_Draw_Rate": "Opp_Prior_Draw_Rate",
            "Prior_Points_Per_Game": "Opp_Prior_Points_Per_Game",
            "Prior_CleanSheet_Rate": "Opp_Prior_CleanSheet_Rate",
            "Prior_FailedScore_Rate": "Opp_Prior_FailedScore_Rate",
            "Prior_Opp_FIFA_Points_Avg": "Opp_Prior_Opp_FIFA_Points_Avg",
            "Prior_Strength_Adjusted_GD_Avg": "Opp_Prior_Strength_Adjusted_GD_Avg",
            "Rest_Days": "Opp_Rest_Days",
            "FIFA_Rank": "Opp_FIFA_Rank",
            "FIFA_Points": "Opp_FIFA_Points",
        }
    )

    data = data.merge(
        opponent,
        on=["Match_ID", "Opponent"],
        how="left",
        validate="one_to_one",
    )

    differences = {
        "FIFA_Points_Diff": ("FIFA_Points", "Opp_FIFA_Points"),
        "FIFA_Rank_Diff": ("FIFA_Rank", "Opp_FIFA_Rank"),
        "Prior_GF_Avg_Diff": ("Prior_GF_Avg", "Opp_Prior_GF_Avg"),
        "Prior_GA_Avg_Diff": ("Prior_GA_Avg", "Opp_Prior_GA_Avg"),
        "Prior_GD_Avg_Diff": ("Prior_GD_Avg", "Opp_Prior_GD_Avg"),
        "Prior_Win_Rate_Diff": ("Prior_Win_Rate", "Opp_Prior_Win_Rate"),
        "Prior_Draw_Rate_Diff": ("Prior_Draw_Rate", "Opp_Prior_Draw_Rate"),
        "Prior_Points_Per_Game_Diff": (
            "Prior_Points_Per_Game",
            "Opp_Prior_Points_Per_Game",
        ),
        "Prior_CleanSheet_Rate_Diff": (
            "Prior_CleanSheet_Rate",
            "Opp_Prior_CleanSheet_Rate",
        ),
        "Prior_FailedScore_Rate_Diff": (
            "Prior_FailedScore_Rate",
            "Opp_Prior_FailedScore_Rate",
        ),
        "Prior_Opp_FIFA_Points_Avg_Diff": (
            "Prior_Opp_FIFA_Points_Avg",
            "Opp_Prior_Opp_FIFA_Points_Avg",
        ),
        "Prior_Strength_Adjusted_GD_Avg_Diff": (
            "Prior_Strength_Adjusted_GD_Avg",
            "Opp_Prior_Strength_Adjusted_GD_Avg",
        ),
        "Rest_Days_Diff": ("Rest_Days", "Opp_Rest_Days"),
    }

    for new_name, (team_value, opponent_value) in differences.items():
        data[new_name] = data[team_value] - data[opponent_value]

    required = [
        "FIFA_Points_Diff",
        "Prior_GF_Avg",
        "Prior_GA_Avg",
        "Prior_GD_Avg",
        "Prior_Win_Rate",
        "Prior_Draw_Rate",
        "Prior_Points_Per_Game",
        "Prior_CleanSheet_Rate",
        "Prior_FailedScore_Rate",
        "Prior_Opp_FIFA_Points_Avg",
        "Prior_Strength_Adjusted_GD_Avg",
        "Rest_Days",
        "Opp_Prior_GF_Avg",
        "Opp_Prior_GA_Avg",
        "Opp_Prior_GD_Avg",
        "Opp_Prior_Win_Rate",
        "Opp_Prior_Points_Per_Game",
        "Opp_Prior_CleanSheet_Rate",
        "Opp_Prior_FailedScore_Rate",
        "Opp_Prior_Opp_FIFA_Points_Avg",
        "Opp_Prior_Strength_Adjusted_GD_Avg",
        "Rest_Days_Diff",
    ]

    if data[required].isna().any().any():
        raise ValueError("Pre-match historical features must not contain missing values.")

    data = data.sort_values(
        ["Date", "Match_ID", "Team"]
    ).reset_index(drop=True)

    if len(data) != 208 or data["Match_ID"].nunique() != 104:
        raise ValueError("Expected 208 rows from 104 matches.")

    if data["Team"].nunique() != 48:
        raise ValueError("Expected 48 teams.")

    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")

    data.to_csv(
        PROCESSED_DIR / "prematch_team_features_208.csv",
        index=False,
    )


def main():
    matches = build_matches()
    team_matches = build_team_match_stats(matches)
    rankings = prepare_rankings(matches)
    all_rankings = pd.read_csv(
        RAW_DIR / "fifa_rankings_11_june_2026.csv"
    )
    pre_tournament = build_pre_tournament_history(
        matches, all_rankings
    )
    build_prematch_features(
        team_matches, rankings, all_rankings, pre_tournament
    )

    print("Data preparation complete")
    print("104 matches")
    print("208 team-match rows")
    print("48 ranked teams")
    print("192 selected pre-tournament history rows (4 per team)")
    print("208 pre-match feature rows")
    print("Historical predictors are complete for all model rows")


if __name__ == "__main__":
    main()
