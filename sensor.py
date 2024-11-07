import numpy as np


class Sensor:
    def __init__(self, mat, sensor, data, labels, interp_funcs):
        self._data = data[f"mat_{mat}"][sensor]
        self._labels = labels[f"mat_{mat}"]
        self._interp_funcs = interp_funcs[f"mat_{mat}"][sensor]

        self._raw_cls_data_list = self._build_raw_cls_data_list()

    def _build_raw_cls_data_list(self):
        cls_data_list = []
        for label in self._labels:
            start = label["start"]
            end = label["end"]
            cls = label["label"]
            target = label["target"]
            heater_data_list = []
            for i in range(10):
                heater_step = self._data[i]
                time_data = heater_step["Time Since PowerOn"].values
                filt_gas = heater_step["Filtered_Gas"].values
                filt_temp = heater_step["Filtered_Temperature"].values
                filt_press = heater_step["Filtered_Pressure"].values
                filt_rh = heater_step["Filtered_Relative_Humidity"].values

                mask = (time_data >= start) & (time_data <= end)
                time_cls = time_data[mask]
                heater_data_list.append({
                    "start": time_cls[0],
                    "end": time_cls[-1],
                    "num_samples": len(time_cls),
                    "sample_times": time_cls,
                    "sample_vals_gas": filt_gas[mask],
                    "sample_vals_temp": filt_temp[mask],
                    "sample_vals_press": filt_press[mask],
                    "sample_vals_rh": filt_rh[mask]
                })

            cls_data_list.append({
                "class": cls,
                "target": target,
                "start": start,
                "end": end,
                "heater_data_list": heater_data_list
            })

        return cls_data_list

    def get_interpolated_data(self, force_num_samples: int = None, include_types: list[str] = ["gas"]):
        cls_data_list = []
        for raw_cls_data in self._raw_cls_data_list:
            cls = raw_cls_data["class"]
            target = raw_cls_data["target"]
            start = raw_cls_data["start"]
            end = raw_cls_data["end"]
            heater_data_list = raw_cls_data["heater_data_list"]

            if force_num_samples:
                num_samples = force_num_samples
            else:
                num_samples = max([el["num_samples"]
                                  for el in heater_data_list])

            sample_times = np.linspace(start, end, num_samples)
            interp_data_list = []
            for include_type in include_types:
                for i in range(10):
                    interp_data = self._interp_funcs[i][include_type](
                        sample_times)
                    interp_data_list.append(interp_data)

            cls_data_list.append({
                "class": cls,
                "target": target,
                "start": start,
                "end": end,
                "time_arr": sample_times,
                "X": np.array(interp_data_list),
                "y": np.array([cls] * num_samples, dtype=np.int32),
                "targets": np.array([target] * num_samples, dtype=np.float32)
            })

        return cls_data_list


def main():
    import pickle
    with open("lpf_sensor_data.pkl", "rb") as f:
        sensor_data = pickle.load(f)

    with open("sensor_labels.pkl", "rb") as f:
        labels = pickle.load(f)

    with open("interpolation_functions.pkl", "rb") as f:
        interp_funcs = pickle.load(f)
    s = Sensor(0, 0, sensor_data, labels, interp_funcs)
    data_list = s.get_interpolated_data(force_num_samples=100, include_types=[
                                        "gas", "temp", "rh", "press"])
    print(data_list)


if __name__ == "__main__":
    main()
