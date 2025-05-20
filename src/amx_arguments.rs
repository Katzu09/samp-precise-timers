use samp::{
    amx::{Allocator, Amx},
    args::Args,
    cell::{Ref, UnsizedBuffer}, // UnsizedBuffer is here
    consts::AmxExecIdx,
    error::AmxError,
    prelude::AmxString,
};
use snafu::{ensure, OptionExt, ResultExt}; // ensure! and context()
use std::{convert::TryInto, num::TryFromIntError};

/// These are the types of arguments the plugin supports for passing on to the callback.
#[derive(Debug, Clone)]
pub enum PassedArgument {
    PrimitiveCell(i32),
    Str(Vec<u8>),
    Array(Vec<i32>),
}

/// A callback which MUST be executed.
/// Its args are already on the AMX stack.
#[ouroboros::self_referencing]
pub(crate) struct StackedCallback {
    pub amx: Amx,
    #[borrows(amx)]
    #[not_covariant]
    pub allocator: Allocator<'this>,
    pub callback_idx: AmxExecIdx,
}

impl StackedCallback {
    #[inline(always)]
    pub fn execute(self) -> Result<i32, AmxError> {
        self.with(|cb| cb.amx.exec(*cb.callback_idx))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct VariadicAmxArguments {
    inner: Vec<PassedArgument>,
}

#[derive(Debug, snafu::Snafu)]
#[snafu(context(suffix(false)))] // This means context selectors are the variant names themselves
pub(crate) enum ArgError {
    #[snafu(display("The list of types ({letters:?}) has {expected} letters, but received {received} arguments."))]
    MismatchedAmountOfArgs {
        received: usize,
        expected: usize,
        letters: Vec<u8>,
    },
    MissingTypeLetters,
    MissingArrayLength,
    MissingArg,
    InvalidArrayLength {
        source: TryFromIntError,
    },
}

impl From<ArgError> for AmxError {
    fn from(value: ArgError) -> Self {
        log::error!("param error: {value:?}");
        AmxError::Params
    }
}

#[rustfmt::skip]
impl VariadicAmxArguments {
    #[cfg(test)]
    pub fn empty() -> Self {
        Self { inner: vec![] }
    }

    fn get_type_letters<const SKIPPED_ARGS: usize>(
        args: &mut Args,
    ) -> Result<impl ExactSizeIterator<Item = u8>, ArgError> {
        let non_variadic_args = SKIPPED_ARGS + 1;
        let letters = args.next::<AmxString>().context(MissingTypeLetters)?.to_bytes();
        let expected = letters.len();
        let received = args.count() - non_variadic_args;
        ensure!(expected == received, MismatchedAmountOfArgs { received, expected, letters });
        Ok(letters.into_iter())
    }

    pub fn from_amx_args<const SKIPPED_ARGS: usize>(
        mut args: Args,
    ) -> Result<VariadicAmxArguments, ArgError> {
        let mut letters = Self::get_type_letters::<SKIPPED_ARGS>(&mut args)?;
        let mut collected_arguments: Vec<PassedArgument> = Vec::with_capacity(letters.len());

        while let Some(type_letter) = letters.next() {
            collected_arguments.push(match type_letter {
                b's' => PassedArgument::Str(args.next::<AmxString>().context(MissingArg)?.to_bytes()),
                b'a' => PassedArgument::Array({
                    ensure!(matches!(letters.next(), Some(b'i' | b'A')), MissingArrayLength);
                    let buffer: UnsizedBuffer = args.next().context(MissingArg)?;
                    let length_ref = *args.next::<Ref<i32>>().context(MissingArg)?;
                    let length = length_ref.try_into().context(InvalidArrayLength)?; // Uses TryFromIntError
                    let sized_buffer = buffer.into_sized_buffer(length);
                    sized_buffer.as_slice().to_vec()
                }),
                _ => PassedArgument::PrimitiveCell(*args.next::<Ref<i32>>().context(MissingArg)?),
            });
        }
        Ok(Self {
            inner: collected_arguments,
        })
    }

    // Added for testing purposes from scheduling.rs
    #[cfg(test)]
    pub(crate) fn get_first_primitive(&self) -> Option<i32> {
        if let Some(PassedArgument::PrimitiveCell(val)) = self.inner.get(0) {
            Some(*val)
        } else {
            None
        }
    }

    pub fn push_onto_amx_stack(
        &self,
        amx: Amx,
        callback_idx: AmxExecIdx,
    ) -> Result<StackedCallback, AmxError> {
        StackedCallback::try_new(amx.clone(), |amx_ref| { 
            let allocator: Allocator = amx_ref.allocator();
            for param in self.inner.iter().rev() {
                match param {
                    PassedArgument::PrimitiveCell(cell_value) => {
                        amx_ref.push(cell_value)?;
                    }
                    PassedArgument::Str(bytes) => {
                        let buffer = allocator.allot_buffer(bytes.len() + 1)?;
                        let amx_str = unsafe { AmxString::new(buffer, bytes) };
                        amx_ref.push(amx_str)?;
                    }
                    PassedArgument::Array(array_cells) => {
                        let amx_buffer = allocator.allot_array(array_cells.as_slice())?;
                        amx_ref.push(array_cells.len())?;
                        amx_ref.push(amx_buffer)?;
                    }
                }
            }
            Ok(allocator)
        }, callback_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Imports ArgError, PassedArgument, VariadicAmxArguments
    use std::collections::VecDeque;
    use std::ptr::NonNull;
    use std::fmt;
    // Note: AmxString, UnsizedBuffer, Ref are already imported via `super::*` from the top level.

    // MockArgType simplified to hold basic Rust types, removing AmxString<'a>
    pub(crate) enum MockArgType {
        StringVal(String), // Stores a Rust String for type 's'
        RefIntVal(i32),    // Stores an i32 for type 'i', 'd', etc. (simulates Ref<i32>)
        ArrayVal(Vec<i32>),// Stores a Vec<i32> for type 'a' (simulates UnsizedBuffer)
    }

    impl fmt::Debug for MockArgType {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                MockArgType::StringVal(s) => f.debug_tuple("StringVal").field(s).finish(),
                MockArgType::RefIntVal(i) => f.debug_tuple("RefIntVal").field(i).finish(),
                MockArgType::ArrayVal(v) => f.debug_tuple("ArrayVal").field(v).finish(),
            }
        }
    }
    
    pub(crate) trait ArgsLike {
        fn next_mock_arg(&mut self) -> Option<MockArgType>;
        fn count(&self) -> usize;
        // To simulate Ref<i32> and UnsizedBuffer, we provide direct pointers to data in TestArgs
        unsafe fn get_current_ref_int_ptr(&mut self) -> Option<NonNull<i32>>;
        unsafe fn get_current_array_val_ptr_len(&mut self) -> Option<(NonNull<i32>, usize)>; // Changed to NonNull<i32>
        fn consume_ref_int_ptr(&mut self); 
        fn consume_array_val_ptr(&mut self);
    }

    #[derive(Default)]
    pub(crate) struct TestArgs {
        args_queue: VecDeque<MockArgType>,
        arg_count: usize,
        // Storages for data that RefIntVal and ArrayVal will point to
        ref_int_storage: Vec<i32>, 
        array_storage: Vec<Vec<i32>>, // Stores Vec<i32> 
        ref_int_cursor: usize,
        array_cursor: usize,
    }

    impl TestArgs {
        pub fn new(arg_count: usize) -> Self {
            TestArgs {
                args_queue: VecDeque::new(),
                arg_count,
                ref_int_storage: Vec::new(),
                array_storage: Vec::new(),
                ref_int_cursor: 0,
                array_cursor: 0,
            }
        }

        pub fn add_string(&mut self, s: &str) {
            self.args_queue.push_back(MockArgType::StringVal(s.to_string()));
        }

        pub fn add_ref_int(&mut self, val: i32) {
            self.ref_int_storage.push(val);
            self.args_queue.push_back(MockArgType::RefIntVal(val));
        }

        pub fn add_array(&mut self, arr: Vec<i32>) { // Renamed from add_unsized_buffer
            self.array_storage.push(arr.clone());
            self.args_queue.push_back(MockArgType::ArrayVal(arr));
        }
    }

    impl ArgsLike for TestArgs {
        fn next_mock_arg(&mut self) -> Option<MockArgType> {
            self.args_queue.pop_front()
        }

        fn count(&self) -> usize {
            self.arg_count
        }

        unsafe fn get_current_ref_int_ptr(&mut self) -> Option<NonNull<i32>> {
            self.ref_int_storage.get_mut(self.ref_int_cursor)
                .map(|val| NonNull::new_unchecked(val as *mut i32))
        }
        
        unsafe fn get_current_array_val_ptr_len(&mut self) -> Option<(NonNull<i32>, usize)> {
             self.array_storage.get_mut(self.array_cursor)
                .map(|arr| {
                    // UnsizedBuffer takes *mut u8, but our PassedArgument::Array is Vec<i32>
                    // The UnsizedBuffer in from_amx_args is converted to Vec<i32>
                    // So for testing, we can provide a *mut i32 and cell count.
                    let ptr = arr.as_mut_ptr();
                    let cell_count = arr.len();
                    (NonNull::new_unchecked(ptr), cell_count)
                })
        }

        fn consume_ref_int_ptr(&mut self) {
            self.ref_int_cursor += 1;
        }

        fn consume_array_val_ptr(&mut self) {
            self.array_cursor += 1;
        }
    }
    
    impl VariadicAmxArguments {
        #[cfg(test)]
        pub fn from_amx_args_test<ArgsType: ArgsLike, const SKIPPED_ARGS: usize>(
            mut args: ArgsType,
        ) -> Result<VariadicAmxArguments, ArgError> {
            // Type string handling
            let type_string_arg = args.next_mock_arg().context(MissingTypeLetters)?;
            let type_string_val = match type_string_arg {
                MockArgType::StringVal(s) => s,
                _ => return Err(ArgError::MissingTypeLetters), // Use ArgError::Variant
            };
            let letters_vec = type_string_val.as_bytes().to_vec();
            
            let non_variadic_args = SKIPPED_ARGS + 1;
            let expected_variadic = letters_vec.len();
            let received_variadic = args.count().saturating_sub(non_variadic_args);
            
            ensure!(expected_variadic == received_variadic, MismatchedAmountOfArgs { 
                received: received_variadic, 
                expected: expected_variadic, 
                letters: letters_vec.clone() 
            });

            let mut letters_iter = letters_vec.into_iter();
            let mut collected_arguments: Vec<PassedArgument> = Vec::with_capacity(expected_variadic);

            while let Some(type_letter) = letters_iter.next() {
                let next_arg_opt = args.next_mock_arg();
                
                collected_arguments.push(match (type_letter, next_arg_opt) {
                    (b's', Some(MockArgType::StringVal(s_val))) => PassedArgument::Str(s_val.as_bytes().to_vec()),
                    (b'a', Some(MockArgType::ArrayVal(arr_val_in_mock))) => { // Matched 'a' and saw ArrayVal
                        ensure!(matches!(letters_iter.next(), Some(b'i' | b'A')), MissingArrayLength);
                        
                        // Simulate getting UnsizedBuffer and Ref<i32> for length
                        // In the test mock, ArrayVal directly gives Vec<i32>
                        // The original from_amx_args would get UnsizedBuffer then convert.
                        // Here, we use the provided Vec<i32> from ArrayVal as the source.
                        // This means we are not testing the UnsizedBuffer -> Vec<i32> conversion here,
                        // but the logic of handling 'a' type specifier.
                        
                        // We still need to simulate the consumption of arguments for count matching
                        // And for the structure of how `a` expects a buffer then a length.
                        let (_array_ptr, _array_cell_count) = unsafe { args.get_current_array_val_ptr_len().context(MissingArg)? };
                        args.consume_array_val_ptr();

                        let len_arg_type = args.next_mock_arg().context(MissingArg)?;
                        let _len_val_from_mock = match len_arg_type { // This is the RefIntVal from mock
                            MockArgType::RefIntVal(val) => val,
                            _ => return Err(ArgError::MissingArg), 
                        };
                        let _ref_int_ptr = unsafe { args.get_current_ref_int_ptr().context(MissingArg)? };
                        args.consume_ref_int_ptr();
                        
                        // The actual PassedArgument::Array will use the arr_val_in_mock directly
                        // The length is implicitly arr_val_in_mock.len() for this simplified test path.
                        // If we wanted to test length mismatch with UnsizedBuffer, this mock would need adjustment.
                        PassedArgument::Array(arr_val_in_mock)
                    },
                    (_, Some(MockArgType::RefIntVal(i_val))) => { // Catches 'i', 'd', 'f', etc.
                        let _ref_int_ptr = unsafe { args.get_current_ref_int_ptr().context(MissingArg)? };
                        args.consume_ref_int_ptr();
                        PassedArgument::PrimitiveCell(i_val)
                    },
                    (_, None) => return Err(ArgError::MissingArg),
                    _ => return Err(ArgError::MissingArg), 
                });
            }
            Ok(Self {
                inner: collected_arguments,
            })
        }
    }

    #[test]
    fn test_from_amx_args_simple_integer() {
        const SKIPPED_ARGS: usize = 3; 
        let mut test_args = TestArgs::new(SKIPPED_ARGS + 1 + 1); // type_str + 1 int
        test_args.add_string("i"); 
        test_args.add_ref_int(123);    

        let result = VariadicAmxArguments::from_amx_args_test::<_, SKIPPED_ARGS>(test_args);
        
        assert!(result.is_ok(), "Result was an error: {:?}", result.err());
        let args_vec = result.unwrap().inner;
        assert_eq!(args_vec.len(), 1);
        match &args_vec[0] {
            PassedArgument::PrimitiveCell(val) => assert_eq!(*val, 123),
            _ => panic!("Expected PrimitiveCell, got {:?}", args_vec[0]),
        }
    }

    #[test]
    fn test_from_amx_args_string() {
        const SKIPPED_ARGS: usize = 3;
        let mut test_args = TestArgs::new(SKIPPED_ARGS + 1 + 1); // type_str + 1 string
        test_args.add_string("s");
        test_args.add_string("hello world");

        let result = VariadicAmxArguments::from_amx_args_test::<_, SKIPPED_ARGS>(test_args);
        assert!(result.is_ok(), "Result was an error: {:?}", result.err());
        let args_vec = result.unwrap().inner;
        assert_eq!(args_vec.len(), 1);
        match &args_vec[0] {
            PassedArgument::Str(s_val) => assert_eq!(*s_val, "hello world".as_bytes()),
            _ => panic!("Expected PassedArgument::Str, got {:?}", args_vec[0]),
        }
    }

    #[test]
    fn test_from_amx_args_array() {
        const SKIPPED_ARGS: usize = 3;
        // type_str ("ai") + array_data + array_length_arg
        let mut test_args = TestArgs::new(SKIPPED_ARGS + 1 + 2); 
        test_args.add_string("ai");
        let array_data = vec![10, 20, 30];
        test_args.add_array(array_data.clone()); // This is for the 'a'
        test_args.add_ref_int(array_data.len() as i32); // This is for the 'i' (length)

        let result = VariadicAmxArguments::from_amx_args_test::<_, SKIPPED_ARGS>(test_args);
        assert!(result.is_ok(), "Result was an error: {:?}", result.err());
        let args_vec = result.unwrap().inner;
        assert_eq!(args_vec.len(), 1);
        match &args_vec[0] {
            PassedArgument::Array(arr_val) => assert_eq!(*arr_val, array_data),
            _ => panic!("Expected PassedArgument::Array, got {:?}", args_vec[0]),
        }
    }

    #[test]
    fn test_from_amx_args_multiple_mixed_args() {
        const SKIPPED_ARGS: usize = 3;
        // type_str ("sisai") + string_arg1 + int_arg + string_arg2 + array_data + array_length_arg
        let mut test_args = TestArgs::new(SKIPPED_ARGS + 1 + 5);
        test_args.add_string("sisai"); // Type string

        test_args.add_string("first string"); // s
        test_args.add_ref_int(42);            // i
        test_args.add_string("second string"); // s
        let array_data = vec![1, 2, 3, 4, 5];
        test_args.add_array(array_data.clone()); // a
        test_args.add_ref_int(array_data.len() as i32); // i (for array length)

        let result = VariadicAmxArguments::from_amx_args_test::<_, SKIPPED_ARGS>(test_args);
        assert!(result.is_ok(), "Result was an error: {:?}", result.err());
        let args_vec = result.unwrap().inner;
        assert_eq!(args_vec.len(), 4); // s, i, s, a

        match &args_vec[0] {
            PassedArgument::Str(s_val) => assert_eq!(*s_val, "first string".as_bytes()),
            _ => panic!("Arg 0 Expected Str, got {:?}", args_vec[0]),
        }
        match &args_vec[1] {
            PassedArgument::PrimitiveCell(i_val) => assert_eq!(*i_val, 42),
            _ => panic!("Arg 1 Expected PrimitiveCell, got {:?}", args_vec[1]),
        }
        match &args_vec[2] {
            PassedArgument::Str(s_val) => assert_eq!(*s_val, "second string".as_bytes()),
            _ => panic!("Arg 2 Expected Str, got {:?}", args_vec[2]),
        }
        match &args_vec[3] {
            PassedArgument::Array(arr_val) => assert_eq!(*arr_val, array_data),
            _ => panic!("Arg 3 Expected Array, got {:?}", args_vec[3]),
        }
    }

    #[test]
    fn test_from_amx_args_mismatched_count_too_few() {
        const SKIPPED_ARGS: usize = 3;
        // Type string "ii", but only one integer provided
        let mut test_args = TestArgs::new(SKIPPED_ARGS + 1 + 1); // types_str + 1 int (instead of 2)
        test_args.add_string("ii");
        test_args.add_ref_int(123);

        let result = VariadicAmxArguments::from_amx_args_test::<_, SKIPPED_ARGS>(test_args);
        assert!(result.is_err());
        match result.err().unwrap() {
            ArgError::MismatchedAmountOfArgs { received, expected, .. } => {
                assert_eq!(received, 1);
                assert_eq!(expected, 2);
            }
            other_err => panic!("Expected MismatchedAmountOfArgs, got {:?}", other_err),
        }
    }

    #[test]
    fn test_from_amx_args_mismatched_count_too_many() {
        const SKIPPED_ARGS: usize = 3;
        // Type string "i", but two integers provided
        let mut test_args = TestArgs::new(SKIPPED_ARGS + 1 + 2); // types_str + 2 ints (instead of 1)
        test_args.add_string("i");
        test_args.add_ref_int(123);
        test_args.add_ref_int(456);

        let result = VariadicAmxArguments::from_amx_args_test::<_, SKIPPED_ARGS>(test_args);
        assert!(result.is_err());
        match result.err().unwrap() {
            ArgError::MismatchedAmountOfArgs { received, expected, .. } => {
                assert_eq!(received, 2);
                assert_eq!(expected, 1);
            }
            other_err => panic!("Expected MismatchedAmountOfArgs, got {:?}", other_err),
        }
    }

    #[test]
    fn test_from_amx_args_empty_type_string() {
        const SKIPPED_ARGS: usize = 3;
        // Type string ""
        let mut test_args = TestArgs::new(SKIPPED_ARGS + 1 + 0); // types_str + 0 args
        test_args.add_string("");

        let result = VariadicAmxArguments::from_amx_args_test::<_, SKIPPED_ARGS>(test_args);
        assert!(result.is_ok(), "Result was an error: {:?}", result.err());
        let args_vec = result.unwrap().inner;
        assert_eq!(args_vec.len(), 0); // Expect no arguments parsed
    }
    
    #[test]
    fn test_from_amx_args_invalid_type_char() {
        const SKIPPED_ARGS: usize = 3;
        // Type string "x", one integer provided. Current logic defaults to PrimitiveCell.
        let mut test_args = TestArgs::new(SKIPPED_ARGS + 1 + 1);
        test_args.add_string("x");
        test_args.add_ref_int(789);

        let result = VariadicAmxArguments::from_amx_args_test::<_, SKIPPED_ARGS>(test_args);
        assert!(result.is_ok(), "Result was an error: {:?}", result.err());
        let args_vec = result.unwrap().inner;
        assert_eq!(args_vec.len(), 1);
        match &args_vec[0] {
            PassedArgument::PrimitiveCell(val) => assert_eq!(*val, 789),
            _ => panic!("Expected PrimitiveCell for unknown type 'x', got {:?}", args_vec[0]),
        }
    }

    #[test]
    fn test_from_amx_args_missing_array_length_specifier() {
        const SKIPPED_ARGS: usize = 3;
        // Type string "a" (missing 'i' or 'A')
        let mut test_args = TestArgs::new(SKIPPED_ARGS + 1 + 1); // types_str + 1 array (no length specifier)
        test_args.add_string("a");
        test_args.add_array(vec![1, 2, 3]); // This argument won't be fully processed

        let result = VariadicAmxArguments::from_amx_args_test::<_, SKIPPED_ARGS>(test_args);
        assert!(result.is_err());
        match result.err().unwrap() {
            ArgError::MissingArrayLength => {} // Correct error
            other_err => panic!("Expected MissingArrayLength, got {:?}", other_err),
        }
    }

    #[test]
    fn test_from_amx_args_missing_arg_for_s() {
        const SKIPPED_ARGS: usize = 3;
        // Type string "s" implies 1 argument after type string.
        // args.count() should reflect this for the initial check to pass.
        let mut test_args = TestArgs::new(SKIPPED_ARGS + 1 + 1); 
        test_args.add_string("s"); 
        // Queue is now empty, so next_mock_arg() for 's' value will be None.

        let result = VariadicAmxArguments::from_amx_args_test::<_, SKIPPED_ARGS>(test_args);
        assert!(result.is_err());
        match result.err().unwrap() {
            ArgError::MissingArg => {} 
            other_err => panic!("Expected MissingArg, got {:?}", other_err),
        }
    }

    #[test]
    fn test_from_amx_args_missing_arg_for_i() {
        const SKIPPED_ARGS: usize = 3;
        // Type string "i" implies 1 argument.
        let mut test_args = TestArgs::new(SKIPPED_ARGS + 1 + 1);
        test_args.add_string("i");
        // Queue is empty for the integer value.

        let result = VariadicAmxArguments::from_amx_args_test::<_, SKIPPED_ARGS>(test_args);
        assert!(result.is_err());
        match result.err().unwrap() {
            ArgError::MissingArg => {}
            other_err => panic!("Expected MissingArg, got {:?}", other_err),
        }
    }
    
    #[test]
    fn test_from_amx_args_missing_buffer_for_a() {
        const SKIPPED_ARGS: usize = 3;
        // Type string "ai" implies 2 arguments (array, then its length).
        // args.count() should reflect this.
        let mut test_args = TestArgs::new(SKIPPED_ARGS + 1 + 2); 
        test_args.add_string("ai");
        // We add the length argument, but not the array data that should precede it.
        test_args.add_ref_int(5); // This will be misinterpreted as the array data.

        let result = VariadicAmxArguments::from_amx_args_test::<_, SKIPPED_ARGS>(test_args);
        assert!(result.is_err());
        match result.err().unwrap() {
            // The current from_amx_args_test logic for 'a' expects MockArgType::ArrayVal.
            // If it gets MockArgType::RefIntVal instead, it hits the fallthrough `_ => return Err(ArgError::MissingArg)`.
            ArgError::MissingArg => {} 
            other_err => panic!("Expected MissingArg (due to type mismatch for buffer), got {:?}", other_err),
        }
    }

    #[test]
    fn test_from_amx_args_missing_length_for_a() {
        const SKIPPED_ARGS: usize = 3;
        // Type string "ai" implies 2 arguments.
        let mut test_args = TestArgs::new(SKIPPED_ARGS + 1 + 2);
        test_args.add_string("ai");
        test_args.add_array(vec![1,2,3]); // Add array data
        // Queue is empty for the length argument.

        let result = VariadicAmxArguments::from_amx_args_test::<_, SKIPPED_ARGS>(test_args);
        assert!(result.is_err());
        match result.err().unwrap() {
            ArgError::MissingArg => {}
            other_err => panic!("Expected MissingArg for length, got {:?}", other_err),
        }
    }
}
