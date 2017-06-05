from collections import Counter

if __name__ == '__main__':
    with open('ensemble_results.txt', 'r') as f:
        corrs = f.readline().strip().split(',')
        original_labels = f.readline().strip().split(',')

    print(len(corrs))
    print(len(original_labels))
    assert len(corrs) == len(original_labels)


    counts = Counter(original_labels)
    correct_counts = {'0': 0, '1': 0, '2': 0}
    for corr, label in zip(corrs, original_labels):
        if corr == '1':
            correct_counts[label] += 1

    print(correct_counts)

