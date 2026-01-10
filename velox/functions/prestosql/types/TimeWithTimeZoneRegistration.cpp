namespace facebook::velox::functions {

namespace {

// Stub: TIME (millis since midnight) → TIME WITH TIMEZONE
void castToTimeWithTimeZone(
    const SimpleVector<int64_t>& inputVector, // TIME is stored as int64_t millis
    exec::EvalCtx& context,
    const SelectivityVector& rows,
    int64_t* rawResults) {
  VELOX_UNSUPPORTED("castToTimeWithTimeZone not yet implemented");
}

} // namespace

void registerTimeWithTimeZoneCasts() {
  exec::registerCast(
      TIME(),
      TIME_WITH_TIMEZONE(),
      castToTimeWithTimeZone);
}

} // namespace facebook::velox::functions
