from itertools import combinations
import networkx as nx
import pandas as pd

from clean_eda import get_cleaned_df


def build_collaboration_graph(df: pd.DataFrame, min_collaborations: int = 1) -> nx.Graph:
    graph = nx.Graph()

    for _, row in df.iterrows():
        people = set()

        for col in ("cast", "director"):
            raw = row.get(col, "")
            if raw:
                for person in raw.split(","):
                    name = person.strip()
                    if name and name.lower() != "unknown":
                        people.add(name)

        for person in people:
            if not graph.has_node(person):
                graph.add_node(person, appearances=0)
            graph.nodes[person]["appearances"] += 1

        for p1, p2 in combinations(sorted(people), 2):
            if graph.has_edge(p1, p2):
                graph[p1][p2]["weight"] += 1
            else:
                graph.add_edge(p1, p2, weight=1)

    if min_collaborations > 1:
        edges_to_remove = [
            (u, v) for u, v, data in graph.edges(data=True) if data.get("weight", 1) < min_collaborations
        ]
        graph.remove_edges_from(edges_to_remove)
        isolates = list(nx.isolates(graph))
        graph.remove_nodes_from(isolates)

    return graph


def summarize_network(graph: nx.Graph, top_n: int = 10) -> None:
    print(f"Nodes (people): {graph.number_of_nodes()} | Edges (collaborations): {graph.number_of_edges()}")
    print(f"Connected components: {nx.number_connected_components(graph)}")

    top_pairs = sorted(
        graph.edges(data=True),
        key=lambda item: item[2].get("weight", 1),
        reverse=True,
    )[:top_n]
    print(f"\n--- Top {top_n} Frequent Collaborator Pairs ---")
    for u, v, data in top_pairs:
        print(f"{u}  <-->  {v}  (shared titles: {data.get('weight', 1)})")

    degree_cent = nx.degree_centrality(graph)
    top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"\n--- Top {top_n} by Degree Centrality ---")
    for person, score in top_degree:
        print(f"{person}: {score:.4f} (neighbors: {graph.degree(person)})")

    # Betweenness on largest connected component for speed
    if graph.number_of_nodes() > 0:
        largest_cc_nodes = max(nx.connected_components(graph), key=len)
        sub_g = graph.subgraph(largest_cc_nodes)
        k_sample = min(250, sub_g.number_of_nodes())
        between_cent = nx.betweenness_centrality(sub_g, k=k_sample, seed=42)
        top_between = sorted(between_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
        print(f"\n--- Top {top_n} Bridge Collaborators (Betweenness Centrality, k={k_sample}) ---")
        for person, score in top_between:
            print(f"{person}: {score:.4f}")


if __name__ == "__main__":
    df = get_cleaned_df()
    g = build_collaboration_graph(df, min_collaborations=1)
    summarize_network(g, top_n=10)