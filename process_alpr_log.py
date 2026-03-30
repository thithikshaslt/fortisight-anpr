import csv

def process_log(csv_path, max_frame_gap=30):
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader]

        if not rows:
            print("CSV is empty.")
            return

        # Ensure rows are sorted by frame (cast frame to int for sorting)
        rows.sort(key=lambda x: int(x['frame']))

        vehicles = [] # List of lists, where each sublist is a vehicle's readings
        current_vehicle = [rows[0]]

        for i in range(1, len(rows)):
            current_frame = int(rows[i]['frame'])
            prev_frame = int(rows[i-1]['frame'])
            
            # If the gap is larger than max_frame_gap, it's a new vehicle
            if (current_frame - prev_frame) > max_frame_gap:
                vehicles.append(current_vehicle)
                current_vehicle = [rows[i]]
            else:
                current_vehicle.append(rows[i])
        
        # Don't forget the last vehicle
        if current_vehicle:
            vehicles.append(current_vehicle)

        print(f"Detected {len(vehicles)} unique vehicles based on frame gaps (>{max_frame_gap} frames).\n")

        best_readings = []
        for vehicle_id, vehicle_rows in enumerate(vehicles):
            # Find the row with the max ocr_conf in this vehicle's rows
            # We need to handle cases where ocr_conf might be empty or invalid floats
            def safe_float(val):
                try: return float(val)
                except: return 0.0

            best_row = max(vehicle_rows, key=lambda x: safe_float(x['ocr_conf']))
            best_readings.append(best_row)

        print("Best readings for each vehicle:")
        print("-" * 60)
        print(f"{'frame':<6} | {'plate':<15} | {'det_conf':<18} | {'ocr_conf'}")
        print("-" * 60)
        for row in best_readings:
            print(f"{row['frame']:<6} | {row['plate']:<15} | {row['det_conf']:<18} | {row['ocr_conf']}")
        print("-" * 60)

        # Save to output file
        output_path = csv_path.replace('.csv', '_optimized.csv')
        # Grab headers from the first row of best_readings (or original rows)
        headers = list(rows[0].keys())
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(best_readings)

        print(f"\nSaved optimized results to: {output_path}")

    except Exception as e:
        print(f"Error processing CSV: {e}")

if __name__ == "__main__":
    process_log('alpr_log.csv', max_frame_gap=50)
