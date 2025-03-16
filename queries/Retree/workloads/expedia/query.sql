EXPLAIN ANALYZE
SELECT
    count(*)
FROM
    S_listings
    JOIN R1_hotels ON S_listings.prop_id = R1_hotels.prop_id
    JOIN R2_searches ON S_listings.srch_id = R2_searches.srch_id
WHERE
    predict (
        '/volumn/Retree_exp/workloads/expedia/model/?.onnx',
        prop_location_score1,
        prop_location_score2,
        -- prop_log_historical_price,
        -- price_usd,
        orig_destination_distance,
        prop_review_score,
        avg_bookings_usd,
        -- stdev_bookings_usd,
        -- position,
        -- prop_country_id,
        count_bookings,
        count_clicks,
        prop_starrating,
        prop_brand_bool
        -- year,
        -- month,
        -- weekofyear,
        -- time,
        -- site_id,
        -- visitor_location_country_id,
        -- srch_destination_id,
        -- srch_length_of_stay,
        -- srch_booking_window,
        -- srch_adults_count,
        -- srch_children_count,
        -- srch_room_count,
        -- srch_saturday_night_bool
        -- random_bool
    ) = ?;