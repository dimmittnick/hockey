Fantasy Points Regression: "nhl_home_fan", "nhl_away_fan"

- 1st Model
- target: "fan_points"
- features = '%shotsOnNetBackhand %shotsOnNetDeflected %shotsOnNetSlap %shotsOnNetSnap %shotsOnNetTipIn %shotsOnNetWrapAround %shotsOnNetWrist homeRoad_H positionCode_C positionCode_D positionCode_R assistsMa7 goalsMa7 plusMinusMa7 pointsMa7 fanPointsMa7 shotsMa7 assistsMa3 goalsMa3 plusMinusMa3 pointsMa3 fanPointsMa3 shotsMa3 assistsLastGame goalsLastGame plusMinusLastGame pointsLastGame fanPointsLastGame shotsLastGame assistsMa10 goalsMa10 plusMinusMa10 pointsMa10 fanPointsMa10 shotsMa10 assistsMa14 goalsMa14 plusMinusMa14 pointsMa14 fanPointsMa14 shotsMa14 timeOnIcePerShiftMa3 timeOnIcePerShiftMa7 timeOnIcePerShiftMa10 timeOnIcePerShiftMa14 timeOnIcePerShiftLastGame ppTimeOnIceMa3 ppTimeOnIceMa7 ppTimeOnIceMa10 ppTimeOnIceMa14 ppTimeOnIceLastGame'.split()

Fantasy Points Regression Scaled: "nhl_home_fan_scale", "nhl_away_fan_scale"

- Scaled version of first model, the output magnitude was much more accurate but do not agree with the names
- target: "fan_points"
- features = '%shotsOnNetBackhand %shotsOnNetDeflected %shotsOnNetSlap %shotsOnNetSnap %shotsOnNetTipIn %shotsOnNetWrapAround %shotsOnNetWrist homeRoad_H positionCode_C positionCode_D positionCode_R assistsMa7 goalsMa7 plusMinusMa7 pointsMa7 fanPointsMa7 shotsMa7 assistsMa3 goalsMa3 plusMinusMa3 pointsMa3 fanPointsMa3 shotsMa3 assistsLastGame goalsLastGame plusMinusLastGame pointsLastGame fanPointsLastGame shotsLastGame assistsMa10 goalsMa10 plusMinusMa10 pointsMa10 fanPointsMa10 shotsMa10 assistsMa14 goalsMa14 plusMinusMa14 pointsMa14 fanPointsMa14 shotsMa14 timeOnIcePerShiftMa3 timeOnIcePerShiftMa7 timeOnIcePerShiftMa10 timeOnIcePerShiftMa14 timeOnIcePerShiftLastGame ppTimeOnIceMa3 ppTimeOnIceMa7 ppTimeOnIceMa10 ppTimeOnIceMa14 ppTimeOnIceLastGame'.split()


Currently using 1st model because it has most believable outputs. Need to introduce more split variables to train new models. 

New Targets:

- overperform dummy
- points dummy
- overperform continuous

- add in goalies
- add in opponents
- try clustering players by similar attributes
- run decision tree to see which features are most important





ohmodel.pkl

- uses one hot encoding and a ton of categorical variables. overfit it
- target: fanPoints
- features: 'savePercLastGame', 'savePercMa3', 'savePercMa7',
       'savePercMa16', 'goalsPerGameLastGame', 'goalsPerGameMa3',
       'goalsPerGameMa7', 'goalsPerGameMa16', 'shotsPerGameLastGame',
       'shotsPerGameMa3', 'shotsPerGameMa7', 'shotsPerGameMa16', 'homeRoadPerf', 'OpHomeDummy', 'OpRoadDummy',
       'OpNowhereDummy','assistsMa7', 'goalsMa7', 'plusMinusMa7', 'pointsMa7',
       'ppPointsMa7', 'fanPointsMa7', 'blockedShotsMa7', 'shootingPctMa7',
       'shotsMa7', 'timeOnIceMa7', 'ppTimeOnIceMa7', 'shiftsMa7',
       'timeOnIcePerShiftMa7', 'assistsMa3', 'goalsMa3', 'plusMinusMa3',
       'pointsMa3', 'ppPointsMa3', 'fanPointsMa3', 'blockedShotsMa3',
       'shootingPctMa3', 'shotsMa3', 'timeOnIceMa3', 'ppTimeOnIceMa3',
       'shiftsMa3', 'timeOnIcePerShiftMa3', 'assistsLastGame', 'goalsLastGame',
       'plusMinusLastGame', 'pointsLastGame', 'ppPointsLastGame',
       'fanPointsLastGame', 'blockedShotsLastGame', 'shootingPctLastGame',
       'shotsLastGame', 'timeOnIceLastGame', 'ppTimeOnIceLastGame',
       'shiftsLastGame', 'timeOnIcePerShiftLastGame', 'assistsMa16',
       'goalsMa16', 'plusMinusMa16', 'pointsMa16', 'ppPointsMa16',
       'fanPointsMa16', 'blockedShotsMa16', 'shootingPctMa16', 'shotsMa16',
       'timeOnIceMa16', 'ppTimeOnIceMa16', 'shiftsMa16',
       'timeOnIcePerShiftMa16', 'avgFanPoints', 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73
 
 
 1.39 mae
 .8348 r^2
 
 This is insane. There is no overfitting or features that could cause this.
 
 params: XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
             grow_policy='depthwise', importance_type=None,
             interaction_constraints='', learning_rate=0.1, max_bin=256,
             max_cat_threshold=64, max_cat_to_onehot=4, max_delta_step=0,
             max_depth=6, max_leaves=0, min_child_weight=1, missing=nan,
             monotone_constraints='()', n_estimators=1000, n_jobs=0,
             num_parallel_tree=1, predictor='auto', random_state=0, ...)

features: 'savePercLastGame', 'savePercMa3', 'savePercMa7',
       'savePercMa16', 'goalsPerGameLastGame', 'goalsPerGameMa3',
       'goalsPerGameMa7', 'goalsPerGameMa16', 'shotsPerGameLastGame',
       'shotsPerGameMa3', 'shotsPerGameMa7', 'shotsPerGameMa16', 'homeRoadPerf', 'OpHomeDummy', 'OpRoadDummy',
       'OpNowhereDummy','assistsMa7', 'goalsMa7', 'plusMinusMa7', 'pointsMa7',
       'ppPointsMa7', 'fanPointsMa7', 'blockedShotsMa7', 'shootingPctMa7',
       'shotsMa7', 'timeOnIceMa7', 'ppTimeOnIceMa7', 'shiftsMa7',
       'timeOnIcePerShiftMa7', 'assistsMa3', 'goalsMa3', 'plusMinusMa3',
       'pointsMa3', 'ppPointsMa3', 'fanPointsMa3', 'blockedShotsMa3',
       'shootingPctMa3', 'shotsMa3', 'timeOnIceMa3', 'ppTimeOnIceMa3',
       'shiftsMa3', 'timeOnIcePerShiftMa3', 'assistsLastGame', 'goalsLastGame',
       'plusMinusLastGame', 'pointsLastGame', 'ppPointsLastGame',
       'fanPointsLastGame', 'blockedShotsLastGame', 'shootingPctLastGame',
       'shotsLastGame', 'timeOnIceLastGame', 'ppTimeOnIceLastGame',
       'shiftsLastGame', 'timeOnIcePerShiftLastGame', 'assistsMa16',
       'goalsMa16', 'plusMinusMa16', 'pointsMa16', 'ppPointsMa16',
       'fanPointsMa16', 'blockedShotsMa16', 'shootingPctMa16', 'shotsMa16',
       'timeOnIceMa16', 'ppTimeOnIceMa16', 'shiftsMa16',
       'timeOnIcePerShiftMa16', 'avgFanPoints', 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73]]

Encoding
cat_vars = ['homeRoad', 'positionCode', 'opponentTeamAbbrev', 'teamAbbrevMerge']

encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encode_cols = pd.DataFrame(encoder.fit_transform(df[cat_vars]))

encode_cols.index = df.index

numer_df = df.drop(cat_vars, axis=1)

encode_df = pd.concat([numer_df, encode_cols], axis=1)
             
model=xgb.XGBRegressor(learning_rate=0.1, n_estimators=1000, max_depth=6)