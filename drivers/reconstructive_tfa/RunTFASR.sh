#! /bin/bash

baselcdir=/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-6
statsdirbase=/nfs/phtess2/ar0/TESS/FFI/LC/FULL/s0006/ISP
periodlist=/nfs/phtess2/ar0/TESS/PROJ/lbouma/cdips/results/cdips_lc_periodfinding/sector-6/initial_period_finding_results.csv
outdir=/nfs/phtess2/ar0/TESS/PROJ/lbouma/CDIPS_LCS/sector-6_TFA_SR

SDETHRESHOLD=10

# Number of phase bins to use in the TFA_SR model
TFASR_NBINS=200

# Fractional difference in rms at which to stop iterating TFA_SR
TFASR_ITERTHRESH=0.001

# Maximum number of TFA_SR iterations to run
TFASR_MAXITER=100

if [ ! -f TFASR_inputlist.txt ] ; then

gawk -v FS=, 'NR > 1 && $5 > '$SDETHRESHOLD' {print $1, $4}' $periodlist | \
    while read starid P ; do
	echo $(find $baselcdir -name '*'$starid'*') $P
    done | \
	gawk '{for(i=1; i <= NF; i += 1) {if($i ~ /.png$/) $i="";} print $0}' \
	     > TFASR_inputlist.txt
fi

rm TFASR_inputlist_?-?.txt

gawk '{split($1,s1,"/");
       if(s1[10] == "cam1_ccd1") print $0 >> "TFASR_inputlist_1-1.txt";
       else if(s1[10] == "cam1_ccd2") print $0 >> "TFASR_inputlist_1-2.txt";
       else if(s1[10] == "cam1_ccd3") print $0 >> "TFASR_inputlist_1-3.txt";
       else if(s1[10] == "cam1_ccd4") print $0 >> "TFASR_inputlist_1-4.txt";
       else if(s1[10] == "cam2_ccd1") print $0 >> "TFASR_inputlist_2-1.txt";
       else if(s1[10] == "cam2_ccd2") print $0 >> "TFASR_inputlist_2-2.txt";
       else if(s1[10] == "cam2_ccd3") print $0 >> "TFASR_inputlist_2-3.txt";
       else if(s1[10] == "cam2_ccd4") print $0 >> "TFASR_inputlist_2-4.txt";
       else if(s1[10] == "cam3_ccd1") print $0 >> "TFASR_inputlist_3-1.txt";
       else if(s1[10] == "cam3_ccd2") print $0 >> "TFASR_inputlist_3-2.txt";
       else if(s1[10] == "cam3_ccd3") print $0 >> "TFASR_inputlist_3-3.txt";
       else if(s1[10] == "cam3_ccd4") print $0 >> "TFASR_inputlist_3-4.txt";
       else if(s1[10] == "cam4_ccd1") print $0 >> "TFASR_inputlist_4-1.txt";
       else if(s1[10] == "cam4_ccd2") print $0 >> "TFASR_inputlist_4-2.txt";
       else if(s1[10] == "cam4_ccd3") print $0 >> "TFASR_inputlist_4-3.txt";
       else if(s1[10] == "cam4_ccd4") print $0 >> "TFASR_inputlist_4-4.txt";
      }' TFASR_inputlist.txt

tmp1=$(mktemp /tmp/RunTFASR.XXXXXX)
tmp2=$(mktemp /tmp/RunTFASR.XXXXXX)


for lclist in TFASR_inputlist_?-?.txt ; do
    camccd=$(echo $lclist | sed -e 's|TFASR_inputlist_||' -e 's|.txt||')

    statsdir=${statsdirbase}_${camccd}-*/stats_files/
    
    gawk '{n = split($1,s1,"/"); split(s1[n],s2,"_"); print s2[1], $0}' $lclist > $tmp1
    gawk '{n = split($1,s1,"/"); split(s1[n],s2,"_"); print s2[1], $0}' ${statsdir}/lc_list_tfa.txt > $tmp2
    grmatch -r $tmp1 -i $tmp2 --match-id --col-inp-id 1 --col-ref-id 1 -o - | \
	gawk '{print $2, $6, $7, $3}' > TFASR_inputlist_${camccd}_xy_period.txt

    templatelists=( ${statsdir}/trendlist_tfa_ap*.txt )
    datelist=${statsdir}/dates_tfa.txt
    
    vartools -l TFASR_inputlist_${camccd}_xy_period.txt -matchstringid \
	 -inputlcformat BGE:BGE:double,BGV:BGV:double,FDV:FDV:double,FKV:FKV:double,FSV:FSV:double,IFE1:IFE1:double,IFE2:IFE2:double,IFE3:IFE3:double,IFL1:IFL1:double,IFL2:IFL2:double,IFL3:IFL3:double,IRE1:IRE1:double,IRE2:IRE2:double,IRE3:IRE3:double,IRM1:IRM1:double,IRM2:IRM2:double,IRM3:IRM3:double,IRQ1:IRQ1:string,IRQ2:IRQ2:string,IRQ3:IRQ3:string,id:RSTFC:string,RSTFC:RSTFC:string,TMID_UTC:TMID_UTC:double,XIC:XIC:double,YIC:YIC:double,CCDTEMP:CCDTEMP:double,NTEMPS:NTEMPS:int,t:TMID_BJD:double,TMID_BJD:TMID_BJD:double,BJDCORR:BJDCORR:double,TFA1:TFA1:double,TFA2:TFA2:double,TFA3:TFA3:double \
	 -expr TFASR1=IRM1 -expr TFASR2=IRM2 -expr TFASR3=IRM3 \
	 -changevariable mag TFASR1 \
	 -changevariable err IRE1 \
	 -TFA_SR ${templatelists[0]} readformat 0 RSTFC IRM1 ${datelist} 20 xycol 2 3 1 0 0 0 ${TFASR_ITERTHRESH} ${TFASR_MAXITER} bin ${TFASR_NBINS} period list column 4 \
	 -changevariable mag TFASR2 \
	 -changevariable err IRE2 \
	 -TFA_SR ${templatelists[1]} readformat 0 RSTFC IRM2 ${datelist} 20 xycol 2 3 1 0 0 0 ${TFASR_ITERTHRESH} ${TFASR_MAXITER} bin ${TFASR_NBINS} period list column 4 \
	 -changevariable mag TFASR3 \
	 -changevariable err IRE3 \
	 -TFA_SR ${templatelists[2]} readformat 0 RSTFC IRM3 ${datelist} 20 xycol 2 3 1 0 0 0 ${TFASR_ITERTHRESH} ${TFASR_MAXITER} bin ${TFASR_NBINS} period list column 4 \
	 -o $outdir columnformat 'RSTFC:Image,TMID_BJD:BJDTDB midexp time,TFASR1:mag,TFASR2:mag,TFASR3:mag' fits copyheader logcommandline -header -numbercolumns -parallel 16 \
	 > TFA_SR_${camccd}.vartools.out

done

rm $tmp1 $tmp2

