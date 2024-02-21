packet_list = ["Packet_2_10", "Packet_3_5","Packet_1_15", "Packet_1_5"]

sorted_list = sorted(packet_list, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))

print(sorted_list)

