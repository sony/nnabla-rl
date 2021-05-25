# DQN (Deep Q-learning Network) reproduction

This reproduction script trains the DQN (Deep Q-learning Network) algorithm proposed by V. Mnih, et al. in the paper: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236).

We tested our implementation with 49 Atari games also used in the [original paper](https://www.nature.com/articles/nature14236) using 3 different initial random seeds:

## Evaluation procedure

We evaluated the algorithm in following settings.

* In every 1M frames (250K steps), the mean reward is evaluated using the Q-Network parameter at that timestep. 
* The evaluation step lasts for 500K frames (125K steps) but the last episode that exceeeds 125K timesteps is not used for evaluation.
* epsilon is set to 0.05 (not greedy).

Mean evaluation score is the mean score among 3 seeds at each iteration.

## Result

Our reproduction results scored higher on 21 out of 49 games and lower on 28 out of 49 games than the original paper.
Most of mean scores are in the range of reported variance and we think that the reproduction results are reasonable.

|Env|nnabla_rl best mean score|Reported score|
|:---|:---:|:---:|
|AlienNoFrameskip-v4|1879.233+/-755.974|**3069+/-1093**|
|AmidarNoFrameskip-v4|608.7+/-292.127|**739.5+/-3024**|
|AssaultNoFrameskip-v4|2164.5+/-640.288|**3359+/-775**|
|AsterixNoFrameskip-v4|4151.902+/-1878.659|**6012+/-1744**|
|AsteroidsNoFrameskip-v4|1118.238+/-455.695|**1629+/-542**|
|AtlantisNoFrameskip-v4|**1585800.0+/-1210026.236**|85641+/-17600|
|BankHeistNoFrameskip-v4|**561.288+/-96.669**|429.7+/-650|
|BattleZoneNoFrameskip-v4|24907.895+/-8244.899|**26300+/-7725**|
|BeamRiderNoFrameskip-v4|**7242.824+/-2421.806**|6846+/-1619|
|BowlingNoFrameskip-v4|41.235+/-17.833|**42.4+/-88**|
|BoxingNoFrameskip-v4|**83.127+/-6.842**|71.8+/-8.4|
|BreakoutNoFrameskip-v4|328.513+/-113.669|**401.2+/-26.9**|
|CentipedeNoFrameskip-v4|4155.931+/-2147.536|**8309+/-5237**|
|ChopperCommandNoFrameskip-v4|1333.333+/-540.57|**6687+/-2916**|
|CrazyClimberNoFrameskip-v4|108337.363+/-18610.659|**114103+/-22797**|
|DemonAttackNoFrameskip-v4|**10466.797+/-5214.039**|9711+/-2406|
|DoubleDunkNoFrameskip-v4|**-12.573+/-3.825**|-18.1+/-2.6|
|EnduroNoFrameskip-v4|**584.907+/-185.919**|301.8+/-24.6|
|FishingDerbyNoFrameskip-v4|**4.692+/-21.284**|-0.8+/-19.0|
|FreewayNoFrameskip-v4|**30.913+/-1.093**|30.3+/-0.7|
|FrostBiteNoFrameskip-v4|**525.464+/-482.123**|328.3+/-250E.5|
|GopherNoFrameskip-v4|5395.422+/-2835.402|**8520+/-3279**|
|GravitarNoFrameskip-v4|**555.6+/-238.396**|306.7+/-223.9|
|HeroNoFrameskip-v4|13973.811+/-1747.23|**19950+/-158**|
|IceHockeyNoFrameskip-v4|-6.972+/-3.433|**-1.6+/-2.5**|
|JamesbondNoFrameskip-v4|540.315+/-674.342|**576.7+/-175.5**|
|KangarooNoFrameskip-v4|**8891.791+/-3184.525**|6740+/-2959|
|KrullNoFrameskip-v4|**6461.866+/-1591.762**|3805+/-1033|
|KungFuMasterNoFrameskip-v4|20993.706+/-9091.718|**23270+/-5955**|
|MontezumaRevengeNoFrameskip-v4|**1.316+/-11.395**|0+/-0|
|MsPacmanNoFrameskip-v4|**2586.658+/-800.127**|2311+/-525|
|NameThisGameNoFrameskip-v4|**7487.895+/-1739.04**|7257+/-547|
|PongNoFrameskip-v4|**19.976+/-1.071**|18.9+/1.3|
|PrivateEyeNoFrameskip-v4|**2327.797+/-5415.25**|1788+/-5473|
|QbertNoFrameskip-v4|7798.325+/-1958.599|**10596+/-3294**|
|RiverRaidNoFrameskip-v4|7512.134+/-1417.171|**8316+/-1049**|
|RoadRunnerNoFrameskip-v4|**40839.61+/-9020.218**|18257+/-4268|
|RobotankNoFrameskip-v4|41.214+/-5.981|**51.6+/-4.7**|
|SeaquestNoFrameskip-v4|3831.481+/-1842.871|**5286+/-1310**|
|SpaceInvadersNoFrameskip-v4|1637.488+/-738.52|**1976+/-893**|
|StarGunnerNoFrameskip-v4|47640.152+/-10720.356|**57997+/-3152**|
|TennisNoFrameskip-v4|**15.796+/-6.604**|-2.5+/-1.9|
|TimePilotNoFrameskip-v4|3701.765+/-1675.655|**5947+/-1600**|
|TutankhamNoFrameskip-v4|184.413+/-30.814|**186.7+/-41.9**|
|UpNDownNoFrameskip-v4|8495.532+/-4267.408|**8546+/-3162**|
|VentureNoFrameskip-v4|262.162+/-355.41|**380.0+/-238.6**|
|VideoPinballNoFrameskip-v4|**236122.36+/-147998.772**|42684+/-16287|
|WizardOfWorNoFrameskip-v4|1667.665+/-866.77|**3393+/-2019**|
|ZaxxonNoFrameskip-v4|2969.154+/-1434.765|**4977+/-1235**|

## Learning curves

|||||
|:---:|:---:|:---:|:---:|
|Alien|Amidar|Assault|Asterix|
|![Alien Result](./reproduction_results/AlienNoFrameskip-v4_results/result.png)|![Amidar Result](./reproduction_results/AmidarNoFrameskip-v4_results/result.png)|![Assault Result](./reproduction_results/AssaultNoFrameskip-v4_results/result.png)|![Asterix Result](./reproduction_results/AsterixNoFrameskip-v4_results/result.png)|
|Asteroids|Atlantis|BankHeist|BattleZone|
|![Asteroids Result](./reproduction_results/AsteroidsNoFrameskip-v4_results/result.png)|![Atlantis Result](./reproduction_results/AtlantisNoFrameskip-v4_results/result.png)|![BankHeist Result](./reproduction_results/BankHeistNoFrameskip-v4_results/result.png)|![BattleZone Result](./reproduction_results/BattleZoneNoFrameskip-v4_results/result.png)|
|BeamRider|Bowling|Boxing|Breakout|
|![BeamRider Result](./reproduction_results/BeamRiderNoFrameskip-v4_results/result.png)|![Bowling Result](./reproduction_results/BowlingNoFrameskip-v4_results/result.png)|![Boxing Result](./reproduction_results/BoxingNoFrameskip-v4_results/result.png)|![Breakout Result](./reproduction_results/BreakoutNoFrameskip-v4_results/result.png)|
|Centipede|ChopperCommand|CrazyClimber|DemonAttack|
|![Centipede Result](./reproduction_results/CentipedeNoFrameskip-v4_results/result.png)|![ChopperCommand Result](./reproduction_results/ChopperCommandNoFrameskip-v4_results/result.png)|![CrazyClimber Result](./reproduction_results/CrazyClimberNoFrameskip-v4_results/result.png)|![DemonAttack Result](./reproduction_results/DemonAttackNoFrameskip-v4_results/result.png)|
|DoubleDunk|Enduro|FishingDerby|Freeway|
|![DoubleDunk Result](./reproduction_results/DoubleDunkNoFrameskip-v4_results/result.png)|![Enduro Result](./reproduction_results/EnduroNoFrameskip-v4_results/result.png)|![FishingDerby Result](./reproduction_results/FishingDerbyNoFrameskip-v4_results/result.png)|![Freeway Result](./reproduction_results/FreewayNoFrameskip-v4_results/result.png)|
|Frostbite|Gopher|Gravitar|Hero|
|![Frostbite Result](./reproduction_results/FrostbiteNoFrameskip-v4_results/result.png)|![Gopher Result](./reproduction_results/GopherNoFrameskip-v4_results/result.png)|![Gravitar Result](./reproduction_results/GravitarNoFrameskip-v4_results/result.png)|![Hero Result](./reproduction_results/HeroNoFrameskip-v4_results/result.png)|
|IceHockey|Jamesbond|Kangaroo|Krull|
|![IceHockey Result](./reproduction_results/IceHockeyNoFrameskip-v4_results/result.png)|![Jamesbond Result](./reproduction_results/JamesbondNoFrameskip-v4_results/result.png)|![Kangaroo Result](./reproduction_results/KangarooNoFrameskip-v4_results/result.png)|![Krull Result](./reproduction_results/KrullNoFrameskip-v4_results/result.png)|
|KungFuMaster|MontezumaRevenge|MsPacman|NameThisGame|
|![KungFuMaster Result](./reproduction_results/KungFuMasterNoFrameskip-v4_results/result.png)|![MontezumaRevenge Result](./reproduction_results/MontezumaRevengeNoFrameskip-v4_results/result.png)|![MsPacman Result](./reproduction_results/MsPacmanNoFrameskip-v4_results/result.png)|![NameThisGame Result](./reproduction_results/NameThisGameNoFrameskip-v4_results/result.png)|
|Pong|PrivateEye|QBert|Riverraid|
|![Pong Result](./reproduction_results/PongNoFrameskip-v4_results/result.png)|![PrivateEye Result](./reproduction_results/PrivateEyeNoFrameskip-v4_results/result.png)|![Qbert Result](./reproduction_results/QbertNoFrameskip-v4_results/result.png)|![Riverraid Result](./reproduction_results/RiverraidNoFrameskip-v4_results/result.png)|
|RoadRunner|Robotank|Seaquest|SpaceInvaders|
|![RoadRunner Result](./reproduction_results/RoadRunnerNoFrameskip-v4_results/result.png)|![Robotank Result](./reproduction_results/RobotankNoFrameskip-v4_results/result.png)|![Seaquest Result](./reproduction_results/SeaquestNoFrameskip-v4_results/result.png)|![SpaceInvaders Result](./reproduction_results/SpaceInvadersNoFrameskip-v4_results/result.png)|
|StarGunner|Tennis|TimePilot|Tutankham|
|![StarGunner Result](./reproduction_results/StarGunnerNoFrameskip-v4_results/result.png)|![Tennis Result](./reproduction_results/TennisNoFrameskip-v4_results/result.png)|![TimePilot Result](./reproduction_results/TimePilotNoFrameskip-v4_results/result.png)|![Tutankham Result](./reproduction_results/TutankhamNoFrameskip-v4_results/result.png)|
|UpNDown|Venture|VideoPinball|WizardOfWor|
|![UpNDown Result](./reproduction_results/UpNDownNoFrameskip-v4_results/result.png)|![Venture Result](./reproduction_results/VentureNoFrameskip-v4_results/result.png)|![VideoPinball Result](./reproduction_results/VideoPinballNoFrameskip-v4_results/result.png)|![WizardOfWor Result](./reproduction_results/WizardOfWorNoFrameskip-v4_results/result.png)|
|Zaxxon||||
|![Zaxxon Result](./reproduction_results/ZaxxonNoFrameskip-v4_results/result.png)||||
