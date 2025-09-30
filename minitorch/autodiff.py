from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_plus_eps = list(vals)
    vals_plus_eps[arg] += epsilon

    vals_minus_eps = list(vals)
    vals_minus_eps[arg] -= epsilon
    return (f(*vals_plus_eps) - f(*vals_minus_eps)) / (2 * epsilon)




variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()
    topological_order = []

    def dfs(variable_: Variable):
        if variable_.unique_id in visited:
            return
        if variable_.is_constant():
            return
        visited.add(variable_.unique_id)
        for parent in variable_.parents:
            dfs(parent)
        topological_order.append(variable_)
    
    dfs(variable)
    topological_order = topological_order[::-1]
    return topological_order # from output to inputs
    




def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    topological_order = topological_sort(variable)

    node_derivative = {}
    node_derivative[variable.unique_id] = deriv

    for cur_node in topological_order:
        cur_deriv = node_derivative.get(cur_node.unique_id, 0.0)

        if not cur_node.is_leaf():
            # cur_node.chain_rule(cur_deriv), cur_node = u + z
            # -> [(u, d(cur_node)/d(u) * cur_deriv ),    (z, d(cur_node)/d(z) * cur_deriv )] 
            for input, deriv_ in cur_node.chain_rule(cur_deriv): 
                prev = node_derivative.get(input.unique_id, 0.0)
                node_derivative[input.unique_id] = prev + deriv_

        else:
            cur_node.accumulate_derivative(cur_deriv)
            continue

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
