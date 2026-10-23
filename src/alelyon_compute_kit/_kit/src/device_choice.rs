//! Which Vulkan device `Context::open` opens: the `ACK_DEVICE` choice.
//!
//! A CPU implementation of Vulkan (Mesa's llvmpipe, Google's SwiftShader) is
//! a compute-capable physical device like any other. On a host whose GPU has
//! no Vulkan driver it is the only one: a Kaggle container, whose runtime
//! mounts no ICD, or a headless VM whose NVIDIA ICD failed to load. Ranking
//! alone used to open it without a word, and every kernel then ran on a CPU
//! emulating a GPU. So the default admits GPUs only, and running on a CPU is
//! something the caller asks for by name:
//!
//! | `ACK_DEVICE`          | admits                                |
//! |-----------------------|---------------------------------------|
//! | unset, empty or `gpu` | discrete, integrated and virtual GPUs |
//! | `cpu`                 | CPU implementations only              |
//! | `any`                 | every compute-capable device          |
//!
//! Each may end in `:<n>` to take the n-th admitted device in the loader's
//! enumeration order instead of the best-ranked one. `gpu:1` is the second
//! GPU, for example one process per card on a two-T4 host. Without an index
//! the rank is the one `open` has always used: a discrete GPU first, then
//! VK_KHR_cooperative_matrix, then VK_KHR_shader_bfloat16, with the first of
//! equals winning.
//!
//! Everything here is pure. The caller enumerates the devices and this
//! module decides, so the policy is tested without a device (tests below).

use ash::vk;

/// The environment variable [`DeviceChoice::from_env`] reads.
pub const ACK_DEVICE_ENV: &str = "ACK_DEVICE";

/// Which kinds of physical device a choice admits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// Discrete, integrated and virtual GPUs. The default.
    Gpu,
    /// CPU implementations of Vulkan only.
    Cpu,
    /// Every compute-capable device, CPU implementations included.
    Any,
}

impl DeviceKind {
    pub fn name(self) -> &'static str {
        match self {
            DeviceKind::Gpu => "gpu",
            DeviceKind::Cpu => "cpu",
            DeviceKind::Any => "any",
        }
    }

    pub fn admits(self, device_type: vk::PhysicalDeviceType) -> bool {
        match self {
            DeviceKind::Gpu => matches!(
                device_type,
                vk::PhysicalDeviceType::DISCRETE_GPU
                    | vk::PhysicalDeviceType::INTEGRATED_GPU
                    | vk::PhysicalDeviceType::VIRTUAL_GPU
            ),
            DeviceKind::Cpu => device_type == vk::PhysicalDeviceType::CPU,
            DeviceKind::Any => true,
        }
    }
}

/// A parsed `ACK_DEVICE` value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceChoice {
    pub kind: DeviceKind,
    /// The n-th admitted device in enumeration order, instead of the
    /// best-ranked one.
    pub index: Option<usize>,
}

impl Default for DeviceChoice {
    fn default() -> Self {
        DeviceChoice {
            kind: DeviceKind::Gpu,
            index: None,
        }
    }
}

impl DeviceChoice {
    /// Parse one value: `gpu`, `cpu` or `any` (any case, surrounding space
    /// ignored), optionally followed by `:<n>` in plain decimal. Empty means
    /// the default. Anything else is refused by name rather than guessed at.
    pub fn parse(text: &str) -> Result<Self, String> {
        let value = text.trim().to_ascii_lowercase();
        if value.is_empty() {
            return Ok(Self::default());
        }
        let refuse = || {
            format!(
                "{ACK_DEVICE_ENV}='{text}' is not gpu, cpu or any, optionally followed by \
                 :<index> (for example gpu:1)"
            )
        };
        let (kind, index) = match value.split_once(':') {
            Some((kind, index)) => (kind, Some(index)),
            None => (value.as_str(), None),
        };
        let kind = match kind {
            "gpu" => DeviceKind::Gpu,
            "cpu" => DeviceKind::Cpu,
            "any" => DeviceKind::Any,
            _ => return Err(refuse()),
        };
        let index = match index {
            None => None,
            // `usize::from_str` takes a leading '+'; an index is digits only
            Some(digits) if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) => {
                Some(digits.parse::<usize>().map_err(|_| refuse())?)
            }
            Some(_) => return Err(refuse()),
        };
        Ok(DeviceChoice { kind, index })
    }

    /// The choice in force: `ACK_DEVICE` if it is set, the default if it is
    /// unset.
    pub fn from_env() -> Result<Self, String> {
        match std::env::var_os(ACK_DEVICE_ENV) {
            None => Ok(Self::default()),
            Some(raw) => match raw.to_str() {
                Some(text) => Self::parse(text),
                None => Err(format!("{ACK_DEVICE_ENV} is not valid UTF-8")),
            },
        }
    }

    /// The value that would select this choice, such as `gpu` or `gpu:1`.
    pub fn describe(&self) -> String {
        match self.index {
            Some(n) => format!("{}:{n}", self.kind.name()),
            None => self.kind.name().to_string(),
        }
    }
}

/// What the choice needs to know about one physical device.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub name: String,
    pub device_type: vk::PhysicalDeviceType,
    /// Whether any queue family supports `COMPUTE`. A device without one is
    /// never admitted.
    pub compute: bool,
    pub cooperative_matrix: bool,
    pub bfloat16: bool,
}

/// The device a choice picked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chosen {
    /// Position in the slice passed to [`choose`], which is the loader's
    /// enumeration order.
    pub index: usize,
    /// One line on how it was chosen, for `DeviceReport::selection`.
    pub selection: String,
}

pub fn type_name(device_type: vk::PhysicalDeviceType) -> &'static str {
    match device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete GPU",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated GPU",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual GPU",
        vk::PhysicalDeviceType::CPU => "CPU",
        _ => "other",
    }
}

/// The rank `open` has always used: discrete +100, cooperative matrices +10,
/// BF16 +1.
fn rank(c: &Candidate) -> i32 {
    let mut score = 0;
    if c.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
        score += 100;
    }
    if c.cooperative_matrix {
        score += 10;
    }
    if c.bfloat16 {
        score += 1;
    }
    score
}

fn listing<'a>(devices: impl Iterator<Item = (usize, &'a Candidate)>) -> String {
    devices
        .map(|(i, c)| format!("[{i}] {} ({})", c.name, type_name(c.device_type)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Pick the device to open, or say why none may be opened.
pub fn choose(candidates: &[Candidate], choice: DeviceChoice) -> Result<Chosen, String> {
    let admitted: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| c.compute && choice.kind.admits(c.device_type))
        .map(|(i, _)| i)
        .collect();
    if admitted.is_empty() {
        return Err(refusal(candidates, choice));
    }
    let skipped = candidates.iter().filter(|c| c.compute).count() - admitted.len();
    let skipped_note = if skipped == 0 {
        String::new()
    } else {
        format!(
            "; {skipped} compute-capable device(s) not admitted: {}",
            listing(
                candidates
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.compute && !choice.kind.admits(c.device_type))
            )
        )
    };
    let choice_text = choice.describe();
    match choice.index {
        Some(n) => match admitted.get(n) {
            Some(&index) => Ok(Chosen {
                index,
                selection: format!(
                    "{ACK_DEVICE_ENV}={choice_text}: admitted device {n} of {} in enumeration \
                     order{skipped_note}",
                    admitted.len()
                ),
            }),
            None => Err(format!(
                "{ACK_DEVICE_ENV}={choice_text} names admitted device {n}, but {} admitted: {}",
                admitted.len(),
                admitted
                    .iter()
                    .enumerate()
                    .map(|(pos, &i)| format!(
                        "{}:{pos} = {}",
                        choice.kind.name(),
                        candidates[i].name
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        },
        None => {
            // strictly greater, so the first of equals keeps its place
            let mut best = admitted[0];
            for &i in &admitted[1..] {
                if rank(&candidates[i]) > rank(&candidates[best]) {
                    best = i;
                }
            }
            Ok(Chosen {
                index: best,
                selection: format!(
                    "{ACK_DEVICE_ENV}={choice_text}: the best-ranked of {} admitted (discrete \
                     GPU, then cooperative matrices, then BF16){skipped_note}",
                    admitted.len()
                ),
            })
        }
    }
}

fn refusal(candidates: &[Candidate], choice: DeviceChoice) -> String {
    let compute: Vec<(usize, &Candidate)> = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| c.compute)
        .collect();
    if compute.is_empty() {
        return format!(
            "no compute-capable device among {}",
            listing(candidates.iter().enumerate())
        );
    }
    let present = listing(compute.iter().copied());
    match choice.kind {
        DeviceKind::Gpu => {
            let has_cpu = compute
                .iter()
                .any(|(_, c)| c.device_type == vk::PhysicalDeviceType::CPU);
            let ask = if has_cpu { "cpu" } else { "any" };
            format!(
                "no GPU among the compute-capable Vulkan devices, only {present}. The kit does \
                 not run on a CPU implementation unless asked: set {ACK_DEVICE_ENV}={ask} to use \
                 it on purpose, or install the GPU's Vulkan driver (ICD) so the loader can see \
                 the GPU"
            )
        }
        DeviceKind::Cpu => format!(
            "no CPU implementation among the compute-capable Vulkan devices ({present}); \
             {ACK_DEVICE_ENV}=cpu needs one, such as Mesa's llvmpipe or SwiftShader"
        ),
        // `any` admits every compute-capable device, and there is one
        DeviceKind::Any => format!("nothing admitted among {present}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::PhysicalDeviceType as T;

    fn dev(name: &str, device_type: T, cooperative_matrix: bool, bfloat16: bool) -> Candidate {
        Candidate {
            name: name.into(),
            device_type,
            compute: true,
            cooperative_matrix,
            bfloat16,
        }
    }

    fn pick(candidates: &[Candidate], value: &str) -> Result<usize, String> {
        choose(candidates, DeviceChoice::parse(value).unwrap()).map(|c| c.index)
    }

    #[test]
    fn unset_and_empty_mean_gpu() {
        let gpu = DeviceChoice {
            kind: DeviceKind::Gpu,
            index: None,
        };
        assert_eq!(DeviceChoice::default(), gpu);
        assert_eq!(DeviceChoice::parse("").unwrap(), gpu);
        assert_eq!(DeviceChoice::parse("  ").unwrap(), gpu);
        assert_eq!(DeviceChoice::parse("gpu").unwrap(), gpu);
    }

    #[test]
    fn names_parse_in_any_case_with_an_optional_index() {
        let cases = [
            ("GPU", DeviceKind::Gpu, None),
            ("cpu", DeviceKind::Cpu, None),
            ("Any", DeviceKind::Any, None),
            (" gpu:1 ", DeviceKind::Gpu, Some(1)),
            ("cpu:0", DeviceKind::Cpu, Some(0)),
            ("any:12", DeviceKind::Any, Some(12)),
        ];
        for (text, kind, index) in cases {
            assert_eq!(
                DeviceChoice::parse(text).unwrap(),
                DeviceChoice { kind, index },
                "{text:?}"
            );
        }
        assert_eq!(DeviceChoice::parse("gpu:1").unwrap().describe(), "gpu:1");
        assert_eq!(DeviceChoice::parse("CPU").unwrap().describe(), "cpu");
    }

    #[test]
    fn a_malformed_value_is_refused_by_name() {
        for bad in [
            "gpu0",
            "gpu:",
            "gpu:x",
            "gpu:-1",
            "gpu:+1",
            "gpu: 1",
            "cpu:1:2",
            "tpu",
            "discrete",
            "gpu:99999999999999999999999999",
        ] {
            let err = DeviceChoice::parse(bad).unwrap_err();
            assert!(err.contains(&format!("'{bad}'")), "{bad}: {err}");
            assert!(err.contains("gpu, cpu or any"), "{bad}: {err}");
        }
    }

    #[test]
    fn a_cpu_implementation_alone_is_refused_by_default_and_named() {
        let kaggle = [dev(
            "llvmpipe (LLVM 15.0.7, 256 bits)",
            T::CPU,
            false,
            false,
        )];
        let err = pick(&kaggle, "").unwrap_err();
        assert!(err.contains("no GPU"), "{err}");
        assert!(
            err.contains("llvmpipe (LLVM 15.0.7, 256 bits) (CPU)"),
            "{err}"
        );
        assert!(err.contains("ACK_DEVICE=cpu"), "{err}");
        assert_eq!(pick(&kaggle, "cpu"), Ok(0), "asking for the CPU opens it");
        assert_eq!(pick(&kaggle, "any"), Ok(0));
    }

    #[test]
    fn the_default_takes_a_gpu_even_where_a_cpu_implementation_outranks_it() {
        // Mesa's lavapipe advertises cooperative matrices; an older
        // integrated GPU may not, so the old rank put the CPU first
        let host = [
            dev("lavapipe", T::CPU, true, false),
            dev("integrated", T::INTEGRATED_GPU, false, false),
        ];
        assert_eq!(pick(&host, ""), Ok(1));
        assert_eq!(pick(&host, "any"), Ok(0), "`any` keeps the old rank");
        assert_eq!(pick(&host, "cpu"), Ok(0));
    }

    #[test]
    fn the_rank_among_gpus_is_unchanged() {
        let host = [
            dev("integrated, coopmat + bf16", T::INTEGRATED_GPU, true, true),
            dev("discrete, bare", T::DISCRETE_GPU, false, false),
            dev("discrete, coopmat", T::DISCRETE_GPU, true, false),
            dev("discrete, coopmat again", T::DISCRETE_GPU, true, false),
        ];
        assert_eq!(
            pick(&host, ""),
            Ok(2),
            "discrete + coopmat, first of equals"
        );
        assert_eq!(pick(&host[..2], ""), Ok(1), "discrete beats extensions");
        let twins = [
            dev("Tesla T4", T::DISCRETE_GPU, true, false),
            dev("Tesla T4", T::DISCRETE_GPU, true, false),
        ];
        assert_eq!(pick(&twins, ""), Ok(0));
    }

    #[test]
    fn an_index_counts_admitted_devices_in_enumeration_order() {
        let host = [
            dev("llvmpipe", T::CPU, false, false),
            dev("Tesla T4", T::DISCRETE_GPU, true, false),
            dev("Tesla T4", T::DISCRETE_GPU, true, false),
        ];
        assert_eq!(pick(&host, "gpu:0"), Ok(1));
        assert_eq!(pick(&host, "gpu:1"), Ok(2));
        assert_eq!(pick(&host, "cpu:0"), Ok(0));
        assert_eq!(pick(&host, "any:0"), Ok(0));
        assert_eq!(pick(&host, "any:2"), Ok(2));
        // an index overrides the rank
        let mixed = [
            dev("integrated", T::INTEGRATED_GPU, false, false),
            dev("discrete", T::DISCRETE_GPU, true, true),
        ];
        assert_eq!(pick(&mixed, "gpu:0"), Ok(0));
    }

    #[test]
    fn an_index_past_the_end_is_refused_with_the_values_that_exist() {
        let twins = [
            dev("Tesla T4", T::DISCRETE_GPU, true, false),
            dev("Tesla T4", T::DISCRETE_GPU, true, false),
        ];
        let err = pick(&twins, "gpu:2").unwrap_err();
        assert!(err.contains("ACK_DEVICE=gpu:2"), "{err}");
        assert!(err.contains("gpu:0 = Tesla T4, gpu:1 = Tesla T4"), "{err}");
        assert!(pick(&twins, "cpu:0")
            .unwrap_err()
            .contains("no CPU implementation"));
    }

    #[test]
    fn a_device_without_a_compute_queue_is_never_admitted() {
        let mut graphics_only = dev("graphics only", T::DISCRETE_GPU, true, true);
        graphics_only.compute = false;
        let host = [
            graphics_only.clone(),
            dev("integrated", T::INTEGRATED_GPU, false, false),
        ];
        assert_eq!(pick(&host, ""), Ok(1));
        assert_eq!(pick(&host, "any"), Ok(1));
        let err = pick(&[graphics_only], "any").unwrap_err();
        assert!(err.starts_with("no compute-capable device"), "{err}");
    }

    #[test]
    fn asking_for_a_cpu_where_there_is_none_is_refused_with_what_is_there() {
        let err = pick(
            &[dev("AMD Radeon RX 9070 XT", T::DISCRETE_GPU, true, true)],
            "cpu",
        )
        .unwrap_err();
        assert!(err.contains("no CPU implementation"), "{err}");
        assert!(
            err.contains("AMD Radeon RX 9070 XT (discrete GPU)"),
            "{err}"
        );
    }

    #[test]
    fn a_virtual_gpu_is_a_gpu_and_other_is_not() {
        assert_eq!(
            pick(&[dev("virtio", T::VIRTUAL_GPU, false, false)], ""),
            Ok(0)
        );
        let err = pick(&[dev("mystery", T::OTHER, false, false)], "").unwrap_err();
        assert!(
            err.contains("ACK_DEVICE=any"),
            "no CPU present, so any: {err}"
        );
        assert_eq!(
            pick(&[dev("mystery", T::OTHER, false, false)], "any"),
            Ok(0)
        );
    }

    #[test]
    fn the_selection_says_which_rule_chose_and_what_it_skipped() {
        let host = [
            dev("llvmpipe", T::CPU, false, false),
            dev("Tesla T4", T::DISCRETE_GPU, true, false),
        ];
        let best = choose(&host, DeviceChoice::default()).unwrap();
        assert!(best
            .selection
            .starts_with("ACK_DEVICE=gpu: the best-ranked of 1 admitted"));
        assert!(
            best.selection.contains("not admitted: [0] llvmpipe (CPU)"),
            "{}",
            best.selection
        );
        let indexed = choose(&host, DeviceChoice::parse("gpu:0").unwrap()).unwrap();
        assert!(indexed
            .selection
            .starts_with("ACK_DEVICE=gpu:0: admitted device 0 of 1"));
        let all = choose(&host, DeviceChoice::parse("any").unwrap()).unwrap();
        assert!(!all.selection.contains("not admitted"), "{}", all.selection);
    }
}
