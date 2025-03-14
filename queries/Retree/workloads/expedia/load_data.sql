CREATE TABLE S_listings AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/expedia/data-extension/?/S_listings.csv', header=True, columns={
    'srch_id': 'INT64',
    'prop_id': 'INT64',
    "position": 'VARCHAR',
    "prop_location_score1": 'FLOAT',
    "prop_location_score2": 'FLOAT',
    "prop_log_historical_price": 'FLOAT',
    "price_usd": 'FLOAT',
    "promotion_flag" : 'INT64',
    "orig_destination_distance" : 'FLOAT'
});

CREATE TABLE R1_hotels AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/expedia/data-extension/?/R1_hotels.csv', header=True, columns={
    'prop_id': 'INT64',
    "prop_country_id": 'VARCHAR',
    "prop_starrating": 'INT64',
    "prop_review_score": 'FLOAT',
    "prop_brand_bool": 'INT64',
    "count_clicks": 'INT64',
    "avg_bookings_usd": 'FLOAT',
    "stdev_bookings_usd": 'FLOAT',
    "count_bookings": 'INT64'
});

CREATE TABLE R2_searches AS
SELECT * FROM read_csv('/volumn/Retree_exp/workloads/expedia/data-extension/?/R2_searches.csv', header=True, columns={
    'srch_id': 'INT64',
    "year": 'VARCHAR',
    "month": 'VARCHAR',
    "weekofyear": 'VARCHAR',
    "time": 'VARCHAR',
    "site_id": 'VARCHAR',
    "visitor_location_country_id": 'VARCHAR',
    "srch_destination_id": 'VARCHAR',
    "srch_length_of_stay": 'INT64',
    "srch_booking_window": 'INT64',
    "srch_adults_count": 'INT64',
    "srch_children_count": 'INT64',
    "srch_room_count": 'INT64',
    "srch_saturday_night_bool": 'INT64',
    "random_bool": 'INT64'
});