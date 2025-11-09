use crate::DeviceSpec;

#[test]
fn test_device_spec_parse() {
    assert_eq!(DeviceSpec::parse("CPU").unwrap(), DeviceSpec::Cpu);
    assert_eq!(DeviceSpec::parse("cpu").unwrap(), DeviceSpec::Cpu);

    #[cfg(feature = "cuda")]
    {
        assert_eq!(DeviceSpec::parse("CUDA:0").unwrap(), DeviceSpec::Cuda { device_id: 0 });
        assert_eq!(DeviceSpec::parse("cuda").unwrap(), DeviceSpec::Cuda { device_id: 0 });
        assert_eq!(DeviceSpec::parse("GPU:2").unwrap(), DeviceSpec::Cuda { device_id: 2 });
    }
}

#[test]
fn test_device_spec_canonicalize() {
    assert_eq!(DeviceSpec::Cpu.canonicalize(), "CPU");

    #[cfg(feature = "cuda")]
    {
        assert_eq!(DeviceSpec::Cuda { device_id: 1 }.canonicalize(), "CUDA:1");
    }
}
