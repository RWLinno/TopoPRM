# DAG case summary

## gsm8k_2184 — gsm8k

- **Question**: Frankie's parents let him have many pets. He has six more snakes than he has cats. He has one less parrot than cats. Six of his pets have four legs. He has 2 dogs. How many pets does he have in total?
- **Gold answer**: 19
- **nodes / edges / orphans**: 4 / 3 / 2
- **q_topo / q_cont**: 0.850 / 1.000

Steps:
  - S0. He has 6 - 2 = <<6-2=4>>4 cats.
  - S1. He has 4 - 1 = <<4-1=3>>3 parrots.
  - S2. He has 4 + 6 = <<4+6=10>>10 snakes.
  - S3. He has a total of 2 + 4 + 3 + 10 = <<2+4+3+10=19>>19 pets.

## math_10559 — math

- **Question**: Let $f : \mathbb{R} \to \mathbb{R}$ be a function such that \[f(xf(y) + x) = xy + f(x)\]for all $x,$ $y.$ Let $n$ be the number of possible values of $f(2),$ and let $s$ be the sum of all possible values of $f(2).$ ...
- **Gold answer**: 0
- **nodes / edges / orphans**: 8 / 13 / 0
- **q_topo / q_cont**: 1.000 / 0.571

Steps:
  - S0. Setting $x = 1$ and $y = -1 - f(1),$ we get
  - S1. \[f(f(-1 - f(1)) +
  - S2. = -1 - f(1) + f(1) = -1.\]Let $a = f(-1 - f(1)) + 1,$ so $f(a) = -1.$
  - S3. Setting $y = a,$ we get
  - S4. \[f(0) = ax + f(x).\]Let $b = f(0),$ so $f(x) = -ax + b.$ Substituting into the given functional equation, we get
  - S5. \[-a(x(-ay + b) + x) + b = xy - ax + b.\]This expands as
  - S6. \[a^2 xy - (ab + a) x + b = xy - ax + b.\]For this to hold for all $x$ and $y,$ we must have $a^2 = 1,$ and $ab + a = a.$ From $a^2 = 1,$ $a = 1$ or $a = -1.$ For either value, $b = 0.$
  - S7. Hence, the solutions are $f(x) = x$ and $f(x) = -x.$ Therefore, $n = 2$ and $s = 2 + (-2) = 0,$ so $n \times s = \boxed{0}.$

## math_4122 — math

- **Question**: What is the sum of all real numbers $x$ that are not in the domain of the function $$f(x) = \frac{1}{x^2-7} + \frac{1}{x^3-8} + \frac{1}{x^4-9}~?$$
- **Gold answer**: 2
- **nodes / edges / orphans**: 5 / 5 / 0
- **q_topo / q_cont**: 1.000 / 0.250

Steps:
  - S0. A real number $x$ is in the domain of $f(x)$ unless $x^2=7$, $x^3=8$, or $x^4=9$.
  - S1. The solutions to $x^2=7$ are $x=\sqrt 7$ and $x=-\sqrt 7$, which sum to $0$.
  - S2. The only solution to $x^3=8$ is $x=2$.
  - S3. The solutions to $x^4=9$ are $x=\sqrt[4]9$ and $x=-\sqrt[4]9$, which sum to $0$.
  - S4. Thus, the sum of all $x$ not in the domain of $f$ is $0+2+0=\boxed{2}$.

## math_2904 — math

- **Question**: A store carries chocolate, vanilla, peppermint, and lemon candies. One day, the store clerk notices that he has fifteen candies total. Furthermore, the number of peppermint and lemon candies together is twice the ...
- **Gold answer**: 1
- **nodes / edges / orphans**: 5 / 4 / 1
- **q_topo / q_cont**: 0.940 / 0.500

Steps:
  - S0. \begin{align*}
  - S1. a+b+c+d &= 15 \\
  - S2. 2(a+b) &= c+d \\
  - S3. c-8 &= d
  - S4. \end{align*} Substituting for $c+d$ in terms of $a+b$ into the first equation gives $3a + 3b = 15$, or $a + b = 5$. This means that $c + d = 10$. The third equation can also be expressed as $c - d ...

## gsm8k_7423 — gsm8k

- **Question**: In the honey shop, the bulk price of honey is $5 per pound and the minimum spend is $40 before tax. The honey is taxed at $1 per pound. If Penny has paid $240 for honey, by how many pounds has Penny’s purchase exceed ...
- **Gold answer**: 32
- **nodes / edges / orphans**: 4 / 3 / 4
- **q_topo / q_cont**: 0.700 / 1.000

Steps:
  - S0. Including tax, a pound of honey costs 5 + 1 = <<5+1=6>>6 dollars
  - S1. The minimum purchase equals 40 / 5 = <<40/5=8>>8 pounds of honey.
  - S2. Penny has bought 240 / 6 = <<240/6=40>>40 pounds of honey
  - S3. Penny has exceeded the minimum purchase by 40 - 8 = <<40-8=32>>32 pounds.

## math_9597 — math

- **Question**: Triangle $ABC$ is a right isosceles triangle. Points $D$, $E$ and $F$ are the midpoints of the sides of the triangle. Point $G$ is the midpoint of segment $DF$ and point $H$ is the midpoint of segment $FE$. What is ...
- **Gold answer**: \frac{5}{11}
- **nodes / edges / orphans**: 4 / 3 / 1
- **q_topo / q_cont**: 0.925 / 0.667

Steps:
  - S0. $\overline{DF}\|\overline{BE}$ and $\overline{DB}\|\overline{FE}$ by the midline theorem and $\angle DBE$ is right, so $DFEB$ is a rectangle. $2BE=BC=AB=2DB$, so $BE=DB$ and $DFEB$ is a square. ...
  - S1. $AB=BC=4x$ and $FG=FH=x$. $\triangle ABC$ has area $\frac{(4x)(4x)}{2}=8x^2$, $\triangle FGH$ has area $\frac{x^2}{2}$, and $\triangle DBE$ has area $\frac{4x^2}{2}=2x^2$. The shaded area is thus ...
  - S2. \frac{\frac{5x^2}{2}}{\frac{11x^2}{2}}=\frac{5x^2}{11x^2}=\boxed{\frac{5}{11}}.
  - S3. \]

## math_8291 — math

- **Question**: What is the maximum number of square inches in the area of a rectangle with a perimeter of 12 inches?
- **Gold answer**: 9
- **nodes / edges / orphans**: 4 / 3 / 1
- **q_topo / q_cont**: 0.925 / 0.667

Steps:
  - S0. Since the perimeter is 12, the sides of the rectangle add up to $12/2 = 6.$ Let $x$ be one side length of the rectangle. Then the other side length is $6 - x,$ so the area is
  - S1. \[x(6 - x) = 6x - x^2.\]Completing the square, we get
  - S2. \[-x^2 + 6x = -x^2 + 6x - 9 + 9 = 9 - (x -
  - S3. ^2.\]Thus, the maximum area of the rectangle is $\boxed{9}$ square inches, which occurs for a $3 \times 3$ square.

## math_5660 — math

- **Question**: Let line $L$ be the intersection of the planes $x + y + z - 6 = 0$ and $2x + 3y + 4z + 5 = 0.$ Find the equation of the plane containing line $L$ and the point $(1,1,1).$ Enter your answer in the form \[Ax + By + Cz ...
- **Gold answer**: 20x + 23y + 26z - 69 = 0
- **nodes / edges / orphans**: 9 / 10 / 2
- **q_topo / q_cont**: 0.933 / 0.500

Steps:
  - S0. Consider the equation
  - S1. \[a(x + y + z -
  - S2. + b(2x + 3y + 4z +
  - S3. = 0,\]where $a$ and $b$ are some real constants. Since $L$ lies in both planes, $L$ satisfies both equations $x + y + z - 6 = 0$ and $2x + 3y + 4z + 5 = 0,$ so $L$ satisfies the equation above.
  - S4. We also want $(1,1,1)$ to satisfy the equation, so we plug in these values, to get
  - S5. \[-3a + 14b = 0.\]We can take $a = 14$ and $b = 3.$ This gives us
  - S6. \[14(x + y + z -
  - S7. + 3(2x + 3y + 4z +
  - S8. = 0,\]which simplifies to $\boxed{20x + 23y + 26z - 69 = 0}.$
