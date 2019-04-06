/* spatial crossmatch w/ ADQL. FAST, if run on gaia archive server */
/* here user_lbouma.dias2014vot is an uploaded table w/ ra,dec in degrees. */
SELECT TOP 500 u.ra, u.dec, u.gmag_estimate, u.ucac_id, g.source_id, DISTANCE(
  POINT('ICRS', u.ra, u.dec),
  POINT('ICRS', g.ra,g.dec)) AS dist,
  g.phot_g_mean_mag as gaia_gmag
FROM user_lbouma.dias2014vot as u, gaiadr2.gaia_source AS g
WHERE 1=CONTAINS(
  POINT('ICRS', u.ra, u.dec),
  CIRCLE('ICRS', g.ra, g.dec, 0.00138888888)
)


/* same, but spatial neighbor search, plus G mag cut */
SELECT TOP 500 u.ra, u.dec, u.gmag_estimate, u.ucac_id, g.source_id, DISTANCE(
  POINT('ICRS', u.ra, u.dec),
  POINT('ICRS', g.ra,g.dec)) AS dist,
  g.phot_g_mean_mag as gaia_gmag
FROM user_lbouma.dias2014vot as u, gaiadr2.gaia_source AS g
WHERE 1=CONTAINS(
  POINT('ICRS', u.ra, u.dec),
  CIRCLE('ICRS', g.ra, g.dec, 0.00138888888)
)
AND
u.gmag_estimate - g.phot_g_mean_mag BETWEEN -2 and 2

