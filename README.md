# hockey
Hockey machine learning model for player performance game by game predictions


Currently makes player by player predictions using xgbRegressor model with features including player goals, points, shots, toi etc. moving averages and numerous OHE variables such as handedness, team, home/away, etc.

The R2 score is .83 and the MAE was 1.2 for the most updated model: ohmodel2020_scaled

This is a very high R2 score for something as random and unpredictable as human performance. The predcitions so far have shown incredible ranking ability, by putting the players with the lowest vegas odds to perform well at the top. Still has the one off questionable decision but when that one player does play well, it makes the model look sharper than vegas.

Uses a robust scaler to deal with outliers and inconsistent magnitudes.

Ultimate goal is to create an interactive website sportsmodels.org that allows users to construct their own predictions by toying with the features and targets and provides them with a downloadable csv. Can be used as tool for fantasy owners or jsut educational purposes. Can be scaled to more sports and can be subscription based. 