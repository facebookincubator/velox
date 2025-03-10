import java.util.Random;

public class FloatGenerator {
  private static final Random random = new Random();

  public static void main(String[] args) {
    if (args.length != 2) {
      System.out.println("usage: java FloatGenerator double|float count");
      System.exit(1);
    }
    String type = args[0];
    int count = Integer.parseInt(args[1]);

    // Detect whether https://bugs.openjdk.org/browse/JDK-4511638 was fixed.
    System.out.println(Double.toString(1.0E23).equals("9.999999999999999E22"));

    if (type.equals("float")) {
      for (int i = 0; i < count; i++) {
        float randomFloat = Float.parseFloat(generateRandom(1, 10, -37, 38));
        System.out.println(randomFloat);
        System.out.println(Float.floatToIntBits(randomFloat));
      }
    } else {
      for (int i = 0; i < count; i++) {
        double randomDouble = Double.parseDouble(generateRandom(1, 18, -307, 308));
        System.out.println(randomDouble);
        System.out.println(Double.doubleToLongBits(randomDouble));
      }
    }
  }

  private static String generateRandom(int minMantissaDigits, int maxMantissaDigits, int minExponent, int maxExponent) {
    int numDigits = minMantissaDigits + random.nextInt(maxMantissaDigits - minMantissaDigits + 1);
    StringBuilder sb = new StringBuilder();

    if (random.nextBoolean()) {
      sb.append('-');
    }

    sb.append(random.nextInt(9) + 1); 
    sb.append('.');

    for (int i = 1; i < numDigits; i++) {
      sb.append(random.nextInt(10));
    }

    int exponent = minExponent + random.nextInt(maxExponent - minExponent + 1);
    sb.append('E');
    sb.append(exponent);

    return sb.toString();
  }
}
