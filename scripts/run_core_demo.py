from microtest.core import add, multiply
# alternatively:
# from microtest import add, multiply


def main():
    print("Running core demo script")

    a = 5
    b = 7

    sum_result = add(a, b)
    product_result = multiply(a, b)

    print(f"Numbers: a={a}, b={b}")
    print(f"Sum: {sum_result}")
    print(f"Product: {product_result}")

    values = [(1, 2), (3, 4), (10, 20)]

    print("\nBatch operations:")
    for x, y in values:
        s = add(x, y)
        p = multiply(x, y)
        print(f"{x}, {y} → sum={s}, product={p}")


if __name__ == "__main__":
    main()