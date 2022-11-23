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