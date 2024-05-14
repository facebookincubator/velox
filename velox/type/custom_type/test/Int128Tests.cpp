#include <memory>
#include <stdexcept>

#include <gtest/gtest.h>
#include <string>
#include "velox/type/custom_type/Int128.h"


namespace facebook::velox::type{


TEST(Int128, objectCreation) {
  
  int128 a = int128(0);
  
  EXPECT_EQ(a.hi(), 0);
  EXPECT_EQ(a.lo(), 0);

  int128 b = int128(23,45);

  EXPECT_EQ(b.hi(), 23);
  EXPECT_EQ(b.lo(), 45);

  int128 c = int128(2147483647, 2147483647);

  EXPECT_EQ(c.hi(), 2147483647);
  EXPECT_EQ(c.lo(), 2147483647);

  int128 d = int128(9223372036854775807, 9223372036854775807);

  EXPECT_EQ(d.hi(), 9223372036854775807);
  EXPECT_EQ(d.lo(), 9223372036854775807);

}

TEST(Int128, bitOR) {
        int128 a = int128(12); //binary 1100
        int128 b = int128(5);  //binary 0101

        EXPECT_EQ(a | b, 13);  //binary 1101

        a |= int128(5);

        EXPECT_EQ(a, 13);

        a = int128(2147483647);

        b = int128(0);

        EXPECT_EQ(a | b, 2147483647);

        int128 c = int128(2147483647, 2147483647);
        int128 d = c | b;
        EXPECT_EQ(d.hi(), 2147483647);
        EXPECT_EQ(d.lo(), 2147483647);

        int128 e = int128(9223372036854775807, 9223372036854775807);
        int128 f = e | b;
        EXPECT_EQ(f.hi(), 9223372036854775807);
        EXPECT_EQ(f.lo(), 9223372036854775807);

}

TEST(Int128, And) {
        int128 a = int128(12); //binary 1100
        int128 b = int128(5);  //binary 0101

        EXPECT_EQ(a & b, 4);   //binary 0100

        a &= int128(5);

        EXPECT_EQ(a, 4);

        a = int128(2147483647);
        b = int128(0);
        EXPECT_EQ(a & b, int128(0));

        int128 c = int128(2147483647, 2147483647);
        int128 d = c & b;
        EXPECT_EQ(d.hi(), 0);
        EXPECT_EQ(d.lo(), 0);

        int128 e = int128(9223372036854775807, 9223372036854775807);
        int128 f = e & b;
        EXPECT_EQ(f.hi(), 0);
        EXPECT_EQ(f.lo(), 0);

        int128 g = e & c;
        EXPECT_EQ(g.hi(), 2147483647);
        EXPECT_EQ(g.lo(), 2147483647);


}

TEST(Int128, LeftShift) {
        int128 a = int128(5); // binary 101

        EXPECT_EQ(a << 1, 10); // binary 1010

        a <<= int128(2);

        EXPECT_EQ(a, 20); //binary 10100

        int128 e = int128(2147483647,2147483647); // binary 1111111111111111111111111111111, 1111111111111111111111111111111
        int128 f = e << 1;
        EXPECT_EQ(f.hi(), 4294967294); // binary 11111111111111111111111111111110
        EXPECT_EQ(f.lo(), 4294967294);

        int128 g = int128(9223372036854775807,9223372036854775807); // binary 111111111111111111111111111111111111111111111111111111111111111,// 111111111111111111111111111111111111111111111111111111111111111
        int128 h = g << 1;
        EXPECT_EQ(h.hi(), -2); // binary 1111111111111111111111111111111111111111111111111111111111111110 with sign
        EXPECT_EQ(h.lo(), 18446744073709551614); //binary 1111111111111111111111111111111111111111111111111111111111111110

}

TEST(Int128, RightShift) {
        int128 a = int128(20); // binary 10100

        EXPECT_EQ(a >> 1, 10); // binary 1010

        a >>= int128(2);

        EXPECT_EQ(a, 5); // binary 101

        int128 e = int128(2147483647,2147483647); // binary 1111111111111111111111111111111, 1111111111111111111111111111111
        int128 f = e >> 1;
        EXPECT_EQ(f.hi(), 1073741823); // binary 0111111111111111111111111111111
        EXPECT_EQ(f.lo(), 9223372037928517631);

        int128 g = int128(9223372036854775807,9223372036854775807); // binary 111111111111111111111111111111111111111111111111111111111111111,// 111111111111111111111111111111111111111111111111111111111111111
        int128 h = g >> 1;
        EXPECT_EQ(h.hi(), 4611686018427387903); // binary 011111111111111111111111111111111111111111111111111111111111111 with sign
        EXPECT_EQ(h.lo(), 13835058055282163711); //binary 101111111111111111111111111111111111111111111111111111111111111


}

TEST(Int128, NOTop) {
        int128 a = int128(5); // binary 0101

        EXPECT_EQ(~a, -6); // binary 1010

        int128 b = int128(2147483647, 2147483647);
        int128 c = ~b;
        EXPECT_EQ(c.hi(), -2147483648);
        EXPECT_EQ(c.lo(), -2147483648);

        int128 d = int128(9223372036854775807, 9223372036854775807);
        int128 e = ~d;
        EXPECT_EQ(e.hi(), -9223372036854775808);
        EXPECT_EQ(e.lo(), 9223372036854775808);

}

TEST(Int128, Equal) {
        int128 a = int128(10);
        int128 b = int128(10);
        EXPECT_EQ(a==b, true);

        b = int128(4);
        EXPECT_EQ(a == b, false);

        a = int128(0);
        b = int128(0);

        EXPECT_EQ(a == b, true);

        int128 c = int128(2147483647, 2147483647);
        int128 d = int128(2147483647, 2147483647);
        EXPECT_EQ(c == d, true);

        d = int128(2147483646, 2147483647);
        EXPECT_EQ(c == d, false);

        c = int128(9223372036854775807, 9223372036854775807);
        d = int128(9223372036854775807, 9223372036854775807);
        EXPECT_EQ(c == d, true);

        d = int128(9223372036854775807, 9223372036854775805);
        EXPECT_EQ(c == d, false);

}

TEST(Int128, NotEqual) {
        int128 a = int128(5);
        int128 b = int128(5);
        EXPECT_EQ(a != b, false);

        b = int128(4);
        EXPECT_EQ(a != b, true);

        int128 c = int128(2147483647, 2147483647);
        int128 d = int128(2147483647, 2147483647);
        EXPECT_EQ(c != d, false);

        d = int128(2147483646, 2147483647);
        EXPECT_EQ(c != d, true);

        c = int128(9223372036854775807, 9223372036854775807);
        d = int128(9223372036854775807, 9223372036854775807);
        EXPECT_EQ(c != d, false);

        d = int128(9223372036854775807, 9223372036854775805);
        EXPECT_EQ(c != d, true);
}

TEST(Int128, objectAdd) {
        int128 a = int128(2);
        int128 b = int128(2);

        EXPECT_EQ(a + b, 4);

        EXPECT_EQ(a += 1, 3);

        EXPECT_EQ(++a,3);

        int128 c = int128(2147483646, 2147483647);
        int128 d = int128(2147483646, 2147483647);
        int128 x = c + d;

        EXPECT_EQ(x.hi(), 4294967292);
        EXPECT_EQ(x.lo(), 4294967294);

        x = c + -d;

        EXPECT_EQ(x.hi(), 0);
        EXPECT_EQ(x.lo(), 0);

        int128 e = int128(4500000000000000000, 4500000000000000000);
        int128 f = int128(4500000000000000000, 4500000000000000000);

        x = e + f;
        EXPECT_EQ(x.hi(), 9000000000000000000);
        EXPECT_EQ(x.lo(), 9000000000000000000);

}

TEST(Int128, objectSub) {
        int128 a = int128(10);
        int128 b = int128(10);

        EXPECT_EQ(a - b, 0);

        int128 c = int128(2147483646, 2147483647);
        int128 d = int128(2147483646, 2147483647);
        int128 x = c - d;

        EXPECT_EQ(x.hi(), 0);
        EXPECT_EQ(x.lo(), 0);

        x = c - -d;

        EXPECT_EQ(x.hi(), 4294967292);
        EXPECT_EQ(x.lo(), 4294967294);

        int128 e = int128(9000000000000000000, 9000000000000000000);
        int128 f = int128(4500000000000000000, 4500000000000000000);

        x = e - f;
        EXPECT_EQ(x.hi(), 4500000000000000000);
        EXPECT_EQ(x.lo(), 4500000000000000000);

}

TEST(Int128, LessThan) {
        int128 a = int128(10);
        int128 b = int128(2);

        EXPECT_EQ(a < b, false);

        a = int128(1);
        EXPECT_EQ(a < b, true);

        b = int128(1);

        EXPECT_EQ(a <= b, true);

        int128 c = int128(2147483646, 2147483647);
        int128 d = int128(1147483646, 2147483647);

        EXPECT_EQ(c < d, false);

        d = int128(3147483646, 2147483647);

        EXPECT_EQ(c < d, true);

        int128 e = int128(9223372036854775807, 9223372036854775807);
        int128 f = int128(9223372036854775807, 9223372036854775807);

        EXPECT_EQ(e <= f, true);

        f = int128(9223372036854775807, 8223372036854775807);

        EXPECT_EQ(e <= f, false);

}

TEST(Int128, GreatThan) {
        int128 a = int128(10);
        int128 b = int128(2);

        EXPECT_EQ(a > b, true);

        int128 d = int128(1);
        EXPECT_EQ(d > b, false);

        EXPECT_EQ(a >= b, true);

        int128 c = int128(10);

        EXPECT_EQ(a >= c, true);

        c = int128(2147483646, 2147483647);
        d = int128(1147483646, 2147483647);

        EXPECT_EQ(c > d, true);

        d = int128(3147483646, 2147483647);

        EXPECT_EQ(c > d, false);

        int128 e = int128(9223372036854775807, 9223372036854775807);
        int128 f = int128(9223372036854775807, 9223372036854775807);

        EXPECT_EQ(e >= f, true);

        f = int128(9223372036854775807, 8223372036854775807);

        EXPECT_EQ(e >= f, true);

}

TEST(Int128, Division) {
        int128 a = int128(0,10);
        int128 b = int128(0,5);

        EXPECT_EQ(a / b, 2);

        a = int128(30);
        b = int128(3);

        EXPECT_EQ(a / b, 10);

        int128 c = int128(9223372036854775807,9223372036854775807);
        int128 d = int128(3147483646, 3147483646);
        
        int128 x = c / d;

        EXPECT_EQ(x.hi(), 0);
        EXPECT_EQ(x.lo(), 2930395539);

        int128 e = int128(9223372036854775807, 9223372036854775807);
        int128 f = int128(6147483646, 6147483646);

        int128 z = e / f;

        EXPECT_EQ(z.hi(), 0);
        EXPECT_EQ(z.lo(), 1500349178);
}

TEST(Int128, Modulo) {
        int128 a = int128(10);
        int128 b = int128(5);

        EXPECT_EQ(a % b, 0);

        int128 c = int128(13);

        EXPECT_EQ(c % b, 3);

        int128 d = int128(9223372036854775807, 9223372036854775807);
        int128 e = int128(3147483646, 3147483646);

        int128 x = d % e;

        EXPECT_EQ(x.hi(), 1540920613);
        EXPECT_EQ(x.lo(), 1540920613);

        int128 f = int128(0, 6223372036854775807);
        int128 g = int128(0, 3147483646);

        int128 z = f % g;

        EXPECT_EQ(z.hi(), 0);
        EXPECT_EQ(z.lo(), 1393216111);

}

TEST(Int128, Multiply) {
        int128 a = int128(10);
        int128 b = int128(5);

        EXPECT_EQ(a * b, 50);

        int128 c = int128(3147483646, 3147483646);
        int128 d = int128(0, 2930395539);

        int128 x = c * d;
        EXPECT_EQ(x.hi(), 9223372035313855194);
        EXPECT_EQ(x.lo(), 9223372035313855194);

        //int128 z = x * x;
        //EXPECT_EQ(z.hi(), 9223372035313855194);
        //EXPECT_EQ(z.lo(), 9223372035313855194);

}

TEST(Int128, toString) {
        int128 b = int128(155);
        std::string test = b.toString(b);
        EXPECT_EQ(test, "155");

        int128 a = int128(200,155);
        test = a.toString(a);
        EXPECT_EQ(test, "3689348814741910323355");

        int128 c = int128(9223372036854775807, 9223372036854775807);
        test = c.toString(c);
        EXPECT_EQ(test, "170141183460469231722463931679029329919");
}

TEST(Int128, mul_overflow) {
        int128 a = int128(9223372036854775807, 9223372036854775807);
        int128 x;
        EXPECT_EQ(int128::mul_overflow(a, a, &x), true);

        int128 c = int128(3147483646, 3147483646);
        int128 d = int128(0, 2930395539);

        int128 z;
        EXPECT_EQ(int128::mul_overflow(c, d, &z), false);

}

TEST(Int128, add_overflow) {
        int128 a = int128(9223372036854775807, 9223372036854775807);
        int128 x;
        EXPECT_EQ(int128::add_overflow(a, a, &x), true);

        int128 c = int128(3147483646, 3147483646);
        int128 d = int128(0, 2930395539);

        int128 z;
        EXPECT_EQ(int128::add_overflow(c, d, &z), false);
}

TEST(Int128, sub_overflow) {
        int128 a = int128(9223372036854775807, 9223372036854775807);
        int128 x;
        EXPECT_EQ(int128::sub_overflow(a, -a, &x), true);

        int128 c = int128(3147483646, 3147483646);
        int128 d = int128(0, 2930395539);

        int128 z;
        EXPECT_EQ(int128::sub_overflow(c, d, &z), false);

        int128 e = int128(1, 1);
        int128 f = int128(9223372036854775807, 9223372036854775807);

        int128 y;
        EXPECT_EQ(int128::sub_overflow(e, -f, &y), true);
}

//TEST(Int128, failTest) {
//  EXPECT_EQ(0, 1);
//}

} // namespace facebook::velox::type