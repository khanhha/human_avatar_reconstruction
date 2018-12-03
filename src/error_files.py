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


ucsc_error_slices  = defaultdict(set)
ucsc_error_slices['UnderBust'] = {'SPRING1894', 'SPRING2391', 'SPRING2622'}
