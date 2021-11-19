# When given a random source_id, it can be part of multiple clusters.  This
# dictionary provides a quick guess at a viable precedence order.
RANKED_MEMBERSHIP_DICT = {
    0.6: {
        'isgroup':[
            'Kraus2014', 'EsplinLuhman2019',  'Furnkranz2019t1',
            'Furnkranz2019t2', 'Meingast2019', 'Goldman2018', 'Rizzuto2017',
            'Damiani2019pms', 'VillaVelez2018', 'Damiani2019ums',
            'Kounkel2018', 'Tian2020', 'Pavlidou2021', 'Ratzenbock2020',
            'RoserSchilbach2020PscEri', 'Meingast2021',
            'Kerr2021', 'CantatGaudin2020a', 'CantatGaudin2020b',
            'CantatGaudin2018a', 'CantatGaudin2019a', 'CastroGinard2020',
            'GaiaCollaboration2018gt250', 'GaiaCollaboration2018lt250',
            'Kounkel2020', 'KounkelCovey2019', 'Ujjwal2020'
        ],
        'isfield':[
            'Gagne2018a', 'Gagne2018b', 'Gagne2018c', 'Gagne2020',
            'Oh2017', 'SIMBAD_PMS', 'SIMBAD_TTS', 'SIMBAD_YSO',
            'SIMBAD_candTTS', 'SIMBAD_candYSO', 'Zari2018pms', 'Zari2018ums',
            'CottenSong2016', 'NASAExoArchive_ps_20210506',
            'HATSandHATNcandidates20210505'
        ]
    }
}
