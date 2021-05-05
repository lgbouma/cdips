# all HATS candidates
mysql -U HSCAND -u hscand_r -phscand_11 -h hatsouth -e 'select HATSname, HATSra, HATSdec, HATS_nam_hat, HATSP, HATSE, HATSq, HATSTODO, HATSclass, HATSFUstat from HATS limit 10000;' -N > hats_candidates.txt

cat hats_candidates.txt | awk '{print $4}' > hats_ids.txt

cat hats_ids.txt | while read HATS_nam_hat; do gaia2read --id $HATS_nam_hat --idtype HAT >> hats_gaia.txt; done

# all HATN candidates
mysql -U HATRED -u hatred_r -phatred11 -h hat -e 'select HTRname, HTRra, HTRdec, HTR_nam_hat, HTRP, HTRE, HTRq, HTRTODO, HTRclass, HTRFUstat from HTR limit 100000;' -N > hatn_candidates.txt

cat hatn_candidates.txt | awk '{print $4}' > hatn_ids.txt

cat hatn_ids.txt | while read HTR_nam_hat; do gaia2read --id $HTR_nam_hat --idtype HAT >> hatn_gaia.txt; done

cat hats_gaia.txt hatn_gaia.txt > hat_gaia_20210505.csv

echo "source_id" > hat_gaia_20210505_cut_source.csv
cat hat_gaia_20210505.csv | awk '{print $1}' >> hat_gaia_20210505_cut_source.csv
sed '/ERROR:/d' ./hat_gaia_20210505_cut_source.csv > hat_gaia_20210505_cut_source_cleaned.csv
