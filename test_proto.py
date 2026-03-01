from protos import fingerprint_pb2


def main():
    fp = fingerprint_pb2.Fingerprint()
    fp.source_file = "example.mid"

    interval = fp.intervals.add()
    interval.value = 4
    interval.duration = 0.5

    # Сериализация
    data = fp.SerializeToString()

    # Запись в файл
    with open("fingerprint.bin", "wb") as f:
        f.write(data)

    print("Fingerprint saved!")

    # Чтение обратно
    fp2 = fingerprint_pb2.Fingerprint()
    with open("fingerprint.bin", "rb") as f:
        fp2.ParseFromString(f.read())

    print("Loaded from file:")
    print(fp2)


if __name__ == "__main__":
    main()
