import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from custom_pca_sffs import CustomPCA, CustomSFFS
from sklearn.linear_model import LogisticRegression

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)

    teams = pd.read_csv("teams.csv")
    ranking = pd.read_csv("ranking.csv")
    players = pd.read_csv("players.csv")

    valid_team_ids = set(df['TEAM_ID_home']).union(set(df['TEAM_ID_away']))
    ranking = ranking[ranking['TEAM_ID'].isin(valid_team_ids)]

    ranking = ranking.drop_duplicates(subset=["TEAM_ID"])

    teams_meta = teams[['TEAM_ID', 'CITY', 'ARENACAPACITY']]
    df = df.merge(teams_meta, left_on='TEAM_ID_home', right_on='TEAM_ID', how='left')
    df.rename(columns={'CITY': 'CITY_home', 'ARENACAPACITY': 'ARENA_home_cap'}, inplace=True)
    df.drop(columns=['TEAM_ID'], inplace=True)

    df = df.merge(teams_meta, left_on='TEAM_ID_away', right_on='TEAM_ID', how='left')
    df.rename(columns={'CITY': 'CITY_away', 'ARENACAPACITY': 'ARENA_away_cap'}, inplace=True)
    df.drop(columns=['TEAM_ID'], inplace=True)

    ranking_home = ranking.rename(columns={
        'TEAM_ID': 'TEAM_ID_home',
        'WINS': 'WINS_home',
        'LOSSES': 'LOSSES_home',
        'RANK': 'RANK_home'
    })
    df = df.merge(ranking_home, on='TEAM_ID_home', how='left')

    ranking_away = ranking.rename(columns={
        'TEAM_ID': 'TEAM_ID_away',
        'WINS': 'WINS_away',
        'LOSSES': 'LOSSES_away',
        'RANK': 'RANK_away'
    })
    df = df.merge(ranking_away, on='TEAM_ID_away', how='left')

    player_count = players.groupby(['TEAM_ID']).size().reset_index(name='NUM_PLAYERS')
    df = df.merge(player_count, left_on='TEAM_ID_home', right_on='TEAM_ID', how='left')
    df.rename(columns={'NUM_PLAYERS': 'NUM_PLAYERS_home'}, inplace=True)
    df.drop(columns=['TEAM_ID'], inplace=True)

    df.fillna(0, inplace=True)

    return df

def normalize_data(X, method):
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'zscore':
        scaler = StandardScaler()
    return scaler.fit_transform(X)

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=None)

def balance_with_smote(X_train, y_train):
    smote = SMOTE()
    return smote.fit_resample(X_train, y_train)

def apply_pca(X, n_components=5):
    pca = CustomPCA(n_components=n_components)
    return pca.fit_transform(X)

def apply_sffs(X, y, k_features=5):
    sffs = CustomSFFS(LogisticRegression(), k_features=k_features)
    return sffs.fit_transform(X, y), sffs
