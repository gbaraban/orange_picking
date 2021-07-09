import rosbag

in_name = "data/flying_data_final_collection_24_feb/2021-02-24-20-44-57.bag"
out_name = "cropped_output.bag"
start_t = 160
end_t = 185
bag_start_t = None
for topic,msg,t in rosbag.Bag(in_name).read_messages():
    bag_start_t = t
    break

with rosbag.Bag(out_name,'w') as outbag:
    for topic,msg,t in rosbag.Bag(in_name).read_messages():
        if "tf" in topic:
            outbag.write(topic,msg,t)
        else:
            local_t = (t - bag_start_t).to_sec()
            if (local_t > start_t) and (local_t < end_t):
                outbag.write(topic,msg,t)
