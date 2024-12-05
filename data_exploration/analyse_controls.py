def nControls(ind):
    a, w, s, d = 0, 0, 0, 0  # Initialize action counters
    for i in ind:  # Loop through each control set in `ind`
        nw, ns, na, nd = i
        w += nw  # Forward
        s += ns  # Back
        a += na  # Left
        d += nd  # Right

    total_actions = a + w + s + d
    proportions = {
        "Forward": w / total_actions if total_actions else 0,
        "Back": s / total_actions if total_actions else 0,
        "Left": a / total_actions if total_actions else 0,
        "Right": d / total_actions if total_actions else 0,
    }

    print("Total Forward:", w)
    print("Total Back:", s)
    print("Total Left:", a)
    print("Total Right:", d)
    print("Total Actions:", total_actions)
    print("Proportions:", proportions)

    return a, w, s, d, total_actions, proportions

def totalControls(indices):
    total_a, total_w, total_s, total_d = 0, 0, 0, 0  # Initialize global counters

    for ind in indices:  # Iterate over all ind datasets
        a, w, s, d = 0, 0, 0, 0  # Initialize local counters for each ind
        for i in ind:  # Loop through controls within each ind
            nw, ns, na, nd = i
            w += nw  # Forward
            s += ns  # Back
            a += na  # Left
            d += nd  # Right
        # Add this ind's totals to the global totals
        total_a += a
        total_w += w
        total_s += s
        total_d += d

    total_actions = total_a + total_w + total_s + total_d

    print("Global Total Forward:", total_w)
    print("Global Total Back:", total_s)
    print("Global Total Left:", total_a)
    print("Global Total Right:", total_d)
    print("Global Total Actions:", total_actions)

    return total_a, total_w, total_s, total_d, total_actions

import matplotlib.pyplot as plt

def plot_controls(indices, labels):
    total_a, total_w, total_s, total_d, total_actions = totalControls(indices)

    # Bar plot values
    actions = ["Left", "Forward", "Back", "Right"]
    totals = [total_a, total_w, total_s, total_d]

    plt.bar(actions, totals, color=['blue', 'green', 'red', 'orange'])

    plt.xlabel("Actions")
    plt.ylabel("Total Counts")
    plt.title("Total Actions by Direction")
    
    for i, value in enumerate(totals):
        plt.text(i, value + 0.5, str(value), ha='center')

    plt.show()








ind0=[[1,0,1,0],[1,0,0,1],[1,0,0,1],[0,0,1,0],[0,0,1,0],[0,0,1,0],[1,0,1,0],[1,0,1,0],[1,0,0,0],[1,0,0,1],[1,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,0],[1,0,1,0],[1,0,1,0],[1,0,0,0]]
ind1=[[1,0,0,0],[0,1,0,1],[0,1,0,1],[1,0,1,0],[1,0,1,0],[1,0,1,0],[1,0,1,0],[1,0,0,0],[1,0,0,0],[1,0,0,1],[1,0,0,1],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,1],[0,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,1,0],[1,0,1,0]]    
ind2=[[1,0,1,0],[0,0,0,1],[0,0,0,1],[0,0,1,0],[0,0,1,0],[0,0,1,0],[1,0,1,0],[1,0,1,0],[1,0,0,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,1,0],[1,0,1,0],[1,0,0,1],[1,0,0,1]]
ind3=[[1,0,0,1],[0,1,0,1],[0,1,1,0],[1,0,1,0],[1,0,1,0],[1,0,1,0],[1,0,1,0],[1,0,0,0],[1,0,0,0],[1,0,0,1],[1,0,0,1],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,1,0],[1,0,1,0],[1,0,0,0]]
ind4=[[1,0,0,0],[0,1,0,1],[0,1,0,1],[1,0,1,0],[1,0,1,0],[1,0,1,0],[1,0,1,0],[1,0,0,0],[1,0,0,0],[1,0,0,1],[1,0,0,1],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,1],[0,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,1,0],[1,0,1,0]]
ind5=[[1,0,1,0],[0,0,0,1],[0,0,0,1],[0,0,1,0],[0,0,1,0],[0,0,1,0],[1,0,1,0],[1,0,1,0],[1,0,0,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,1,0],[1,0,1,0],[1,0,0,1],[1,0,0,1]]
ind6=[[1,0,0,1],[0,1,0,1],[0,1,1,0],[1,0,1,0],[1,0,1,0],[1,0,1,0],[1,0,1,0],[1,0,0,0],[1,0,0,0],[1,0,0,1],[1,0,0,1],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,1,0],[1,0,1,0],[1,0,0,0]]
ind7=[[1,0,0,0],[0,1,0,1],[0,1,0,1],[1,0,1,0],[1,0,1,0],[1,0,1,0],[1,0,1,0],[1,0,0,0],[1,0,0,0],[1,0,0,1],[1,0,0,1],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,1],[0,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,1,0],[1,0,1,0]]
ind8=[[1,0,1,0],[0,0,0,1],[0,0,0,1],[0,0,1,0],[0,0,1,0],[0,0,1,0],[1,0,1,0],[1,0,1,0],[1,0,0,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,0],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,1,0],[1,0,1,0],[1,0,0,1],[1,0,0,1]]
ind9=[[1,0,0,1],[0,1,0,1],[0,1,1,0],[1,0,1,0],[1,0,1,0],[1,0,1,0],[1,0,1,0],[1,0,0,0],[1,0,0,0],[1,0,0,1],[1,0,0,1],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,1,0],[1,0,1,0],[1,0,0,0]]


# Example usage for all indices
indices = [ind0, ind1, ind2, ind3, ind4, ind5, ind6, ind7, ind8, ind9]
labels = ["ind0", "ind1", "ind2", "ind3", "ind4", "ind5", "ind6", "ind7", "ind8", "ind9"]
for i, ind in enumerate(indices):
    print(f"\nMetrics for ind{i}:")
    nControls(ind)

totalControls(indices)
plot_controls(indices, labels)