'''
Purpose: To calculate PPR or contact graph enhanced edges, used for constructing residue-level graph structure.
Input: Protein contact map or adjacency matrix.
Output: Index and weights of enhanced edges of PPR.
'''
import numpy as np
from tqdm import tqdm


def get_ppr_matrix(adj_matrix: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes, dtype=adj_matrix.dtype)   # 加自环
    D_tilde = np.diag(1.0 / np.sqrt(A_tilde.sum(axis=1) + 1e-12))
    H = D_tilde @ A_tilde @ D_tilde
    # PPR closed-form
    return alpha * np.linalg.inv(np.eye(num_nodes, dtype=adj_matrix.dtype) - (1 - alpha) * H)

def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
    A = A.copy()  # RNAsmol 原实现会原地修改，保险起见 copy
    num_nodes = A.shape[0]
    col_idx = np.arange(num_nodes)

    if k >= num_nodes:
        # 不需要裁剪，只做按行归一化
        norm = A.sum(axis=1)
        norm[norm <= 0] = 1.0
        return A / norm[:, None]

    # 按行排序：对每一行 i，找到该行从小到大的列索引
    # 把每行中“除 top-k 最大值以外”的元素置 0
    # A[col_idx, A.argsort(axis=1)[:, :num_nodes - k]] = 0.0
    row_idx = np.arange(num_nodes)[:, None]  # (N,1)
    cols_to_zero = A.argsort(axis=1)[:, :num_nodes - k]  # (N, N-k)

    A[row_idx, cols_to_zero] = 0.0  # 形状可广播到 (N, N-k)

    # 按行归一化
    norm = A.sum(axis=1)
    norm[norm <= 0] = 1.0
    return A / norm[:, None]

def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
    A = A.copy()
    A[A < eps] = 0.
    norm = A.sum(axis=1)
    norm[norm <= 0] = 1.0
    return A / norm[:, None]

def contacts_to_adj_matrix(contacts, num_nodes: int, remove_self_loops: bool = True) -> np.ndarray:
    """
    contacts: 形如 [(i,j), ...]，你当前代码是 j>=i，且包含 (i,i)
    返回: 对称的 0/1 邻接矩阵 adj (对角线为0，若 remove_self_loops=True)
    """
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for (i, j) in contacts:
        if remove_self_loops and i == j:
            continue
        if i < 0 or j < 0 or i >= num_nodes or j >= num_nodes:
            continue
        # 无向边补成对称
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return adj

def ppr_edges_from_contacts(
    contacts,
    num_nodes: int,
    alpha: float = 0.1,
    k: int = 5,
    eps: float = None,
    remove_self_loops_in_adj: bool = True
):
    """
    输出 RNAsmol 风格的:
      edge_index: [2, E] (numpy int64)
      edge_attr : [E]    (numpy float32)
    """
    adj = contacts_to_adj_matrix(contacts, num_nodes, remove_self_loops=remove_self_loops_in_adj)

    ppr = get_ppr_matrix(adj, alpha=alpha)

    if k is not None:
        ppr = get_top_k_matrix(ppr, k=k)
    elif eps is not None:
        ppr = get_clipped_matrix(ppr, eps=eps)
    else:
        raise ValueError("k 和 eps 至少提供一个（与 RNAsmol 一致）。")

    # RNAsmol 的抽边方式：ppr[i,j] > 0 即为边，权重为 ppr[i,j]
    edge_index, edge_attr = [], []
    for i, row in enumerate(ppr):
        js = np.where(row > 0)[0]
        for j in js:
            edge_index.append((i, j))
            edge_attr.append(ppr[i, j])

    edge_index = np.array(edge_index, dtype=np.int64)   # [E, 2]
    edge_attr  = np.array(edge_attr, dtype=np.float32)  # [E]
    return edge_index, edge_attr


def main():
    adj_file = "./data/SHS50K/adj_matrix.npy"
    protein_edge = np.load(adj_file, allow_pickle=True)
    
    all_edge_index = []
    all_edge_attr = []

    for idx, contacts in tqdm(enumerate(protein_edge), total=len(protein_edge), desc="PPR构图"):
        num_nodes = int(np.max(contacts)) + 1

        
        edge_index, edge_attr = ppr_edges_from_contacts(
            contacts=contacts,
            num_nodes=num_nodes,
            alpha=0.1,
            k=5,            # 对应 RNAsmol 的 self.k
            eps=None
        )
        all_edge_index.append(edge_index) 
        all_edge_attr.append(edge_attr)

    # 保存
    np.save("./data/SHS50k/edge_index_ppr.npy", np.array(all_edge_index, dtype=object))
    np.save("./data/SHS50k/edge_attr_ppr.npy", np.array(all_edge_attr, dtype=object))


if __name__ == "__main__":
    main()