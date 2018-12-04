from collections import defaultdict

mpii_error_slices = defaultdict(set)
mpii_error_slices['Bust'] = {'nl_6019a', 'nl_5595a', 'csr4239a', 'CSR0094A', 'CSR1289A','CSR1226A', 'CSR1334A', 'CSR1622A', \
                             'CSR1742A', 'CSR2705A', 'CSR2781A', 'CSR2825A', 'CSR2833A', 'CSR2834A', 'CSR2845A', 'CSR2928A'}

mpii_error_slices['Aux_UnderBust_Bust_0'] = {'csr4793a','CSR0017A','CSR1178A','CSR1444A','CSR1473A','CSR1604A','CSR1721A','CSR1827A','CSR1906A','CSR2149A','CSR2300A',
                                             'CSR2603A','CSR2825A','CSR2834A','CSR2862A','CSR2997A','CSR3010A','CSR3027A','nl_1088a','nl_1229a''nl_5218a','nl_5949a'}

mpii_error_slices['UnderCrotch'] = {'csr4148a', 'csr4150a', 'csr4175a', 'csr4125a',
                                    'csr4285a', 'csr4293a', 'csr4339a', 'csr4376a', 'csr4388', 'csr4406a', 'csr4428a', 'csr4453a', 'csr4472a', 'csr4544a', 'csr4582a',
                                    'csr4610', 'csr4766a', 'CSR0627A', 'CSR1083A', 'CSR1310A', 'CSR1311A', 'CSR1315A', 'CSR1345A', 'CSR1356A', 'CSR1390A', 'CSR1403A',
                                    'CSR1448A', 'CSR1451A', 'CSR1549A', 'CSR1563A', 'CSR1566A', 'CSR1623A', 'CSR1686A', 'CSR1722A', 'CSR1742A', 'CSR1849A', 'CSR2022A',
                                    'CSR2023A', 'CSR2724A', 'nl_1079a', 'nl_1144a', 'nl_5960a', 'nl_7023a'}

mpii_error_slices['Aux_Knee_UnderCrotch_3'] = {'csr4024a', 'csr4038a', 'csr4096a', 'csr4137a', 'csr4150a', 'csr4175a', 'csr4192a', 'csr4293a', 'csr4393a',
                                            'csr4406a', 'csr4426a', 'csr4440a', 'csr4453a', 'csr4471a', 'csr4479a', 'csr4483a', 'csr4502a', 'csr4506a',
                                            'csr4536a', 'csr4540a', 'csr4544a', 'csr4582a', 'csr4610a', 'csr4612a', 'csr4656a', 'csr4662a', 'csr4744a',
                                            'csr4763a', 'csr4771a', 'CSR0166A', 'CSR0257A', 'CSR0307A', 'CSR0381A', 'CSR0461A', 'CSR0468A', 'CSR0627A',
                                            'CSR1180A', 'CSR1303A', 'CSR1310A', 'CSR1331A', 'CSR1345A', 'CSR1390A', 'CSR1458A', 'CSR1648A', 'CSR1703A',
                                            'CSR1722A', 'CSR1793A', 'CSR1822A', 'CSR1849A', 'CSR1867A', 'CSR1923A', 'CSR1995A', 'CSR2020A', 'CSR2023A',
                                            'CSR2075A', 'CSR2078A', 'CSR2082A', 'CSR2089A', 'CSR2093A', 'CSR2117A', 'CSR2150A', 'CSR2280A', 'CSR2302A',
                                            'CSR2326A', 'CSR2380A', 'CSR2380A', 'CSR2402A', 'CSR2422A', 'CSR2581A', 'CSR2643A', 'CSR2691A', 'CSR2708A',
                                            'CSR2724A', 'CSR2800A', 'CSR2826A', 'CSR2996A', 'CSR3001A', 'nl_2198a', 'nl_5219a', 'nl_5250a', 'nl_5561a',
                                            'nl_6481a', 'nl_6486a', 'nl_6571a'}

mpii_error_slices['Aux_Knee_UnderCrotch_3'] = {'csr4062a', 'csr4120a', 'csr4140a', 'csr4150a', 'csr4178a', 'csr4250a', 'csr4260a', 'csr4292a', 'csr4364a', 'csr4370a', 'csr4393a',
                                               'csr4439a', 'csr4441a', 'csr4479a', 'csr4507a', 'csr4532a', 'csr4538a', 'csr4544a', 'csr4548a', 'csr4578a', 'csr4601a', 'csr4604a',
                                               'csr4610a', 'csr4645a', 'csr4646a', 'csr4656a', 'csr4662a', 'csr4670a', 'csr4695a', 'csr4737a', 'CSR0084A', 'CSR0143A', 'CSR0163A',
                                               'CSR0189A', 'CSR0213A', 'CSR0235A', 'CSR0269A', 'CSR0407A', 'CSR0468A', 'CSR0472A', 'CSR0499A', 'CSR0544A', 'CSR0627A', 'CSR0640A',
                                               'CSR0668A', 'CSR1035A', 'CSR1140A', 'CSR1279A', 'CSR1285A', 'CSR1290A', 'CSR1296A', 'CSR1303A', 'CSR1310A', 'CSR1316A', 'CSR1345A',
                                               'CSR1390A', 'CSR1608A', 'CSR1703A', 'CSR1722A', 'CSR1728A', 'CSR1751A', 'CSR1753A', 'CSR1772A', 'CSR1796A', 'CSR1827A', 'CSR1832A',
                                               'CSR1846A', 'CSR1849A', 'CSR1854A', 'CSR1862A', 'CSR1990A', 'CSR1995A', 'CSR2012A', 'CSR2016A', 'CSR2023A', 'CSR2039A', 'CSR2075A',
                                               'CSR2140A', 'CSR2295A', 'CSR2361A', 'CSR2371A', 'CSR2382A', 'CSR2431A', 'CSR2490A', 'CSR2636A', 'CSR2643A', 'CSR2691A', 'CSR2716A',
                                               'CSR2731A', 'CSR2745A', 'CSR2800A', 'CSR2826A', 'CSR2828A', 'CSR2943A', 'CSR2957A', 'CSR3001A', 'CSR3018A', 'CSR3028A',
                                               'nl_1054a', 'nl_1128a', 'nl_1146a', 'nl_1173a', 'nl_1294a', 'nl_1368a', 'nl_1427a', 'nl_2375a', 'nl_5214a', 'nl_5219a', 'nl_5323a',
                                               'nl_5482a', 'nl_5690a', 'nl_5753a', 'nl_5755a', 'nl_5776a', 'nl_5780a', 'nl_5841a', 'nl_6121a', 'nl_6571a', 'nl_6582a', 'nl_6699a',
                                               'nl_6778a', 'nl_6818a'}

ucsc_error_slices  = defaultdict(set)
ucsc_error_slices['UnderBust'] = {'SPRING1894', 'SPRING2391', 'SPRING2622'}
