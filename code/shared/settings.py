"""Shared parameters and frozen verification values; no file exports.
INPUT: constants preserved from the former configuration/figure-manifest files.
OUTPUT: in-memory settings for independently executed result scripts.
Frozen values are used for verification only, never substituted for calculations.
"""

DEFINITIONS = {'CONFIG': {'dates': {'last_pow': '2022-09-14',
                      'merge': '2022-09-15',
                      'pre_start': '2022-09-07',
                      'stable_start': '2022-11-13',
                      'stable_end': '2023-02-11',
                      'var_start': '2018-01-01',
                      'var_end': '2021-10-31'},
            'legacy': {'eth_price': 0.05,
                       'etc_price': 0.1,
                       'overhead': 1.33,
                       'cutoff': 1.0,
                       'eth_block_seconds': 15,
                       'etc_block_seconds': 13,
                       'best_efficiency': 2.1667},
            'extension': {'price': 0.1, 'chains': ['ETC', 'ETHW', 'RVN', 'ERG']},
            'uq': {'price': 0.1,
                   'overhead': 1.33,
                   'seed': 20260921,
                   'draws': 20000,
                   'memory_GB': 5.0,
                   'memory_rule': 'strict',
                   'price_triangular': [0.03, 0.1, 0.12],
                   'overhead_triangular': [1.05, 1.33, 1.5],
                   'scenarios': [{'name': 'Prices and overhead; equal hash weights',
                                  'key': 'price_overhead',
                                  'beta': None,
                                  'kappa': None,
                                  'rho': 0},
                                 {'name': 'Composition only; mean ASIC share 25%',
                                  'key': 'composition_only',
                                  'beta': [1, 3],
                                  'kappa': 100,
                                  'rho': 0,
                                  'fixed_econ': True},
                                 {'name': 'Joint; mean ASIC share 10%',
                                  'key': 'joint_asic10',
                                  'beta': [1, 9],
                                  'kappa': 100,
                                  'rho': 0},
                                 {'name': 'Joint; mean ASIC share 25%',
                                  'key': 'joint_asic25',
                                  'beta': [1, 3],
                                  'kappa': 100,
                                  'rho': 0},
                                 {'name': 'Joint; mean ASIC share 50%',
                                  'key': 'joint_asic50',
                                  'beta': [1, 1],
                                  'kappa': 100,
                                  'rho': 0},
                                 {'name': 'Joint; diffuse within-class shares',
                                  'key': 'joint_diffuse',
                                  'beta': [1, 3],
                                  'kappa': 20,
                                  'rho': 0},
                                 {'name': 'Joint; correlated price and overhead',
                                  'key': 'joint_correlated',
                                  'beta': [1, 3],
                                  'kappa': 100,
                                  'rho': 0.6},
                                 {'name': 'Joint; allow recorded 5 GB devices',
                                  'key': 'joint_memory5',
                                  'beta': [1, 3],
                                  'kappa': 100,
                                  'rho': 0,
                                  'memory_rule': 'allow_5GB'}]}},
 'INPUT_CHECKSUMS': {'data/extension/inputs/crypto_daily.csv': '726b23139b5dcb636e380d3a232f2aa504c7dd64787b826bef0ef58b7e5fcb9c',
                     'data/extension/inputs/erg_revenue.csv': '63d7318893737754cafdb14b2069c891cbfa8db216e1e5eb0d74fe19e03e28c8',
                     'data/extension/inputs/etc_revenue.csv': '316d3566697ed58aa1fcee6dd2e9d7f0feabcad813f32b88c87ed846eefd9e51',
                     'data/extension/inputs/ethw_revenue.csv': 'beb68af621ac62d5fb24772b0fe08d706a183da7d41067ff7b43b65fb1c0dd6f',
                     'data/extension/inputs/rvn_revenue.csv': '2a6a9e24568c95414955805ce18e64812036c3ebb4e9c60c5fe26f6d00861fb5',
                     'data/original/Coin_Metrics_Network_Data_2023-03-14T12-53.xlsx': 'e26ff2f7c6108931c5c0437d52017913910608b1be489adbe815001b3b3f5dec',
                     'data/original/Ethereum+.xlsx': 'b5cc5bfc06a860855081810913095e2a133103127c2c48e32f0b86ab4e19901e',
                     'data/original/EthereumHardware.xlsx': 'f3feec386317f71c7135881499cba3d1fd682b77a100f252c77f221738a94c62',
                     'data/original/EthereumHardwareGPU.xlsx': '0932815ca8f664954a26cfab2572361ca4feca2607ccffad7cd8be2a3ed45d34'},
 'FROZEN_REFERENCE': {'monte_carlo': 'scenario,key,draws,valid_draws,invalid_pct,pre_GW_p025,pre_GW_median,pre_GW_p975,post_budget_GW_p025,post_budget_GW_median,post_budget_GW_p975,saving_at_budget_pct_p025,saving_at_budget_pct_median,saving_at_budget_pct_p975,realized_ASIC_hash_share_p025,realized_ASIC_hash_share_median,realized_ASIC_hash_share_p975\n'
                                     'Prices and overhead; equal hash '
                                     'weights,price_overhead,20000,20000,0.0,2.5978659356025005,3.0244743152953935,3.69948325145982,0.18499091130138642,0.24337373175972105,0.4900876355016408,86.56843804077675,91.78163725078002,93.73798057937414,0.056411063138572096,0.16350448974996878,0.19703988287687574\n'
                                     'Composition only; mean ASIC share '
                                     '25%,composition_only,20000,20000,0.0,2.1507834961854737,2.851133686185737,3.2596364408863137,0.20997988649840987,0.20997988649840987,0.20997988649840987,90.23705142926812,92.63521428115797,93.55818078772744,0.09423985518597529,0.2748863861588099,0.745462608854369\n'
                                     'Joint; mean ASIC share '
                                     '10%,joint_asic10,20000,20000,0.0,2.49736769388255,3.0156143635215806,3.8055127518733314,0.1854224574188714,0.24336883562810335,0.48769263476916985,86.54971020007514,91.6997850128812,93.89671487679841,0.00869019553346186,0.14258758194327267,0.39497555535343304\n'
                                     'Joint; mean ASIC share '
                                     '25%,joint_asic25,20000,20000,0.0,2.2491291197498735,2.8913011534687314,3.9953401341492443,0.18540461110406495,0.24391772443473703,0.4923464304976135,86.49714945814215,91.21791950817538,93.66990367479595,0.030821912719880898,0.2612172883680215,0.7321456558245243\n'
                                     'Joint; mean ASIC share '
                                     '50%,joint_asic50,20000,20000,0.0,1.9164642106924723,2.6573926050848806,4.397402608911736,0.18536143365625216,0.24407368160459753,0.4990660124755521,85.78488641623817,90.24103126152633,93.37448398189292,0.09076688200947156,0.5422051640457207,0.9810992815731088\n'
                                     'Joint; diffuse within-class '
                                     'shares,joint_diffuse,20000,20000,0.0,2.1098886146354765,2.8855728124495306,4.068021295741645,0.18528508044365766,0.24449803729132077,0.4885632928445552,85.79995571747315,91.12994167543582,93.93639789411709,0.01724217788704721,0.2570003128343231,0.7635695081451347\n'
                                     'Joint; correlated price and '
                                     'overhead,joint_correlated,20000,20000,0.0,2.3073850706285453,2.904669498899651,3.9722000550900702,0.18531226239359008,0.24284688062264004,0.4939343262075127,86.21468375264048,91.29492060629794,93.96169257348834,0.02926036947617038,0.26020443179530095,0.7291502316279262\n'
                                     'Joint; allow recorded 5 GB '
                                     'devices,joint_memory5,20000,20000,0.0,2.2055664488498894,2.8987534878591736,3.877116047642605,0.18522587345898325,0.2430246769584586,0.4906597752655388,86.27997520259697,91.18270747556102,93.77499326266953,0.02971115405146766,0.26313378174055746,0.7360381763046671\n'}}

ORIGINAL_FIGURES = {'Figure_Upload/HashrateETHETC_Fig_2.png': {'source': 'Energy_Demand_ETH_1013/Figure_Upload/HashrateETHETC_Fig_2.png',
                                            'sha256': 'cb19ea4633e1d18f4f2b02ebff66d2d6642fc51a074699f7e136035d15518223'},
 'Figure_Upload/fig_energy_td_vs_bu_Fig_6.png': {'source': 'Energy_Demand_ETH_1013/Figure_Upload/fig_energy_td_vs_bu_Fig_6.png',
                                                 'sha256': 'eafa285c984c0ad41fea640cd4549e404735ed1ec0eca47768f47b77113a51e5'},
 'Figure_Upload/fig_profitability_threshold_Fig_7.png': {'source': 'Energy_Demand_ETH_1013/Figure_Upload/fig_profitability_threshold_Fig_7.png',
                                                         'sha256': 'd09d752bbda2eaafb366018bbe66e098573412ed84afd714062cfddf06be4349'},
 'Figure_Upload/ETCEnergy_Fig_9.png': {'source': 'Energy_Demand_ETH_1013/Figure_Upload/ETCEnergy_Fig_9.png',
                                       'sha256': 'a6d799454854f4b2875f448259cdc6e500f0ae6384f43299293eac5acf7a2deb'},
 'Figure_Upload/ext_var_irf_Fig_11.png': {'source': 'Energy_Demand_ETH_1013/Figure_Upload/ext_var_irf_Fig_11.png',
                                          'sha256': '679bfdbb7d2e21c8d61880d632d32dd2bbd67c8f611739c1124b01c4ab73cbb6'},
 'Figure/PriceETHETC.png': {'source': 'Energy_Demand_ETH_1013/PriceETHETC.png',
                            'sha256': 'af46ce463b9a449fcd7d7363dc394aecff52ac2f14e4327a619ff9aa1fe8ed4b'},
 'Figure/fig_hashrate_migration.png': {'source': 'Energy_Demand_ETH_1013/fig_hashrate_migration.png',
                                       'sha256': 'eed8f5a334f5a4b9fad8e5ecdc80acfa180c1b90e92423b278d270465c9f342d'},
 'Figure/fig_energy_bottomup.png': {'source': 'Energy_Demand_ETH_1013/fig_energy_bottomup.png',
                                    'sha256': '49f9ec34299825274347638ec1c8de59b148b267eed3cb73a0cf9cbc6917d466'},
 'Figure/fig_cohort_split.png': {'source': 'Energy_Demand_ETH_1013/fig_cohort_split.png',
                                 'sha256': 'bd2b6742d9ad598da03455525becb2ffde6268a0536a6f371274ede5ffae67ef'},
 'Figure/ext_sensitivity_ce_series.png': {'source': 'Energy_Demand_ETH_1013/Figures/extended/ext_sensitivity_ce_series.png',
                                          'sha256': '64974e84ffbbe38446f23791907995238a71b04ddafca2f77c03ed8f2112e759'},
 'Figure/ext_sensitivity_ce_bar.png': {'source': 'Energy_Demand_ETH_1013/Figures/extended/ext_sensitivity_ce_bar.png',
                                       'sha256': 'b89e4b7f2e313b4b708b85ed59683c79e7c849771c4f8234e9bd0f1a98639ad6'},
 'Figure/fig_panel_legends_bottom.png': {'source': 'Energy_Demand_ETH_1013/Figures/extended/fig_panel_legends_bottom.png',
                                         'sha256': '7f65fd8a1b65982964606df352202882b8583969db0cabcfb12edce455bcb627'},
 'Figure/ext_var_fevd.png': {'source': 'Energy_Demand_ETH_1013/Figures/extended/ext_var_fevd.png',
                             'sha256': 'e132f2b34c7d945dee5502415abc035b640fb6a53c4997401db6ed56e8e5ccee'},
 'Figure/ext_var_irf_BIC.png': {'source': 'Energy_Demand_ETH_1013/Figures/extended/ext_var_irf.png',
                                'sha256': '679bfdbb7d2e21c8d61880d632d32dd2bbd67c8f611739c1124b01c4ab73cbb6'}}
