#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace TAT {
	using Rank = unsigned int;
	using Size = unsigned long;

	namespace legs {
		/**
		 * class Legs is to identify a leg with its id.
		 *
		 * Legs(string) create new legs with next id,
		 * or return the legs created with same name.
		 * Legs(IdType) to specify its id directly,
		 * this method will NOT maintain id and name map.
		 */
		class Legs {
			public:
				using IdType = int;
				IdType id = -1;
				Legs() = default;
				explicit Legs(IdType i) : id{i} {}
				explicit Legs(const std::string& name) {
					try {
						id = name2id.at(name);
					} catch (const std::out_of_range& e) {
						id = total++;
						name2id[name] = id;
						id2name[id] = name;
					} // exsit name
				}
				static IdType total;
				static std::map<std::string, IdType> name2id;
				static std::map<IdType, std::string> id2name;
		}; // class Legs

		Legs::IdType Legs::total = 0;
		std::map<std::string, Legs::IdType> Legs::name2id = {};
		std::map<Legs::IdType, std::string> Legs::id2name = {};

		bool operator==(const Legs& a, const Legs& b) {
			return a.id == b.id;
		}
		bool operator!=(const Legs& a, const Legs& b) {
			return a.id != b.id;
		}
		bool operator<(const Legs& a, const Legs& b) {
			return a.id < b.id;
		}

		std::ostream& operator<<(std::ostream& out, const Legs& value) {
			try {
				return out << Legs::id2name.at(value.id);
			} catch (const std::out_of_range& e) {
				return out << "UserDefinedLeg" << value.id;
			}
		}
	} // namespace legs
	using legs::Legs;

	//
	//      L      EEEEE   GGGG    SSS          N    N    AA    M     M  EEEEE
	//      L      E      G    G  S   S         N    N   A  A   MM   MM  E
	//      L      E      G       S             NN   N  A    A  M M M M  E
	//      L      E      G       S             N N  N  A    A  M  M  M  E
	//      L      EEEE   G        SSS          N  N N  A    A  M     M  EEEE
	//      L      E      G  GGG      S         N   NN  AAAAAA  M     M  E
	//      L      E      G    G      S         N    N  A    A  M     M  E
	//      L      E      G   GG  S   S         N    N  A    A  M     M  E
	//      LLLLL  EEEEE   GGG G   SSS   _____  N    N  A    A  M     M  EEEEE
	//
	/**
	 * namespace legs_name containt 190 predefined legs.
	 *
	 * 190 predefined legs, including Tmp0~99,
	 * and (Phy, 8 direction) * 0~9 (90 totaly).
	 */
	namespace legs_name {
#define TAT_DefineLeg(x) static const TAT::Legs x(#x)
#define TAT_DefineLegs(n)                       \
		TAT_DefineLeg(Phy##n);                      \
		TAT_DefineLeg(Left##n);                     \
		TAT_DefineLeg(Right##n);                    \
		TAT_DefineLeg(Up##n);                       \
		TAT_DefineLeg(Down##n);                     \
		TAT_DefineLeg(LeftUp##n);                   \
		TAT_DefineLeg(LeftDown##n);                 \
		TAT_DefineLeg(RightUp##n);                  \
		TAT_DefineLeg(RightDown##n)
#define TAT_Legs                                \
		TAT_DefineLegs();                           \
		TAT_DefineLegs(1);                          \
		TAT_DefineLegs(2);                          \
		TAT_DefineLegs(3);                          \
		TAT_DefineLegs(4);                          \
		TAT_DefineLegs(5);                          \
		TAT_DefineLegs(6);                          \
		TAT_DefineLegs(7);                          \
		TAT_DefineLegs(8);                          \
		TAT_DefineLegs(9)
		TAT_Legs;
#undef TAT_Legs
#undef TAT_DefineLegs
#define TAT_DefineLegs(n)                       \
		TAT_DefineLeg(Leg##n##0);                   \
		TAT_DefineLeg(Leg##n##1);                   \
		TAT_DefineLeg(Leg##n##2);                   \
		TAT_DefineLeg(Leg##n##3);                   \
		TAT_DefineLeg(Leg##n##4);                   \
		TAT_DefineLeg(Leg##n##5);                   \
		TAT_DefineLeg(Leg##n##6);                   \
		TAT_DefineLeg(Leg##n##7);                   \
		TAT_DefineLeg(Leg##n##8);                   \
		TAT_DefineLeg(Leg##n##9)
#define TAT_Legs                                \
		TAT_DefineLegs();                           \
		TAT_DefineLegs(1);                          \
		TAT_DefineLegs(2);                          \
		TAT_DefineLegs(3);                          \
		TAT_DefineLegs(4);                          \
		TAT_DefineLegs(5);                          \
		TAT_DefineLegs(6);                          \
		TAT_DefineLegs(7);                          \
		TAT_DefineLegs(8);                          \
		TAT_DefineLegs(9)
		TAT_Legs;
#undef TAT_Legs
#undef TAT_DefineLegs
#undef TAT_DefineLeg
	} // namespace legs_name

	namespace tensor{
		template<class T>
			class Tensor {
				public:
					Rank rank;
					Size size;
					std::vector<Size> dims;
					std::vector<Legs> legs;
					std::vector<T> data;

					template<class A=std::vector<Size>, class B=std::vector<Legs>>
						Tensor(A&& a, B&& b) :
							rank(a.size()),
							size(std::accumulate(a.begin(), a.end(), Size(1), std::multiplies<Size>())),
							dims(std::forward<A>(a)),
							legs(std::forward<B>(b)),
							data(size)
				{}

					auto get_index(const std::vector<Size>& position) const {
						Size index = 0;
						for(int i=0; i<rank; i++){
							index = index*dims[i] + position[i];
						}
						return index;
					}

					auto get_position(const std::map<Legs, Size>& dict) const {
						std::vector<Size> res;
						std::transform(legs.begin(), legs.end(), std::back_inserter(res), [&dict](const Legs& l){return dict.at(l);});
						return res;
					}

					const T& operator()(const std::vector<Size>& position) const {
						return data[get_index(position)];
					}
					T& operator()(const std::vector<Size>& position) {
						return data[get_index(position)];
					}
					const T& operator()(const std::map<Legs, Size>& dict) const {
						return data[get_index(get_position(dict))];
					}
					T& operator()(const std::map<Legs, Size>& dict) {
						return data[get_index(get_position(dict))];
					}

					void generate(std::function<T()> g) {
						std::generate(data.begin(), data.end(), g);
					}

					void inplace_op_unary(std::function<T(T)> f){
						std::transform(data.begin(), data.end(), data.begin(), f);
					}

					template<class T2>
					auto outplace_op_unary(std::function<T2(T)> f){
						auto res = Tensor<T2>(dims, legs);
						std::transform(data.begin(), data.end(), res.data.begin(), f);
						return res;
					}

					template<class T2>
					void inplace_op_binary(std::function<T(T, T2)> f, const Tensor<T2>& t2) {
						std::transform(data.begin(), data.end(), t2.data.begin(), data.begin(), f);
					}

					template<class T1, class T2, class T>
					auto outplace_op_binary(std:function<T(T1, T2)> f, const Tensor<T1>& t1, const Tensor<T2>& t2) {
						auto res = Tensor<T>(t1.dims, t1.legs);
						std::transform(t1.data.begin(), t1.data.end(), t2.data.begin(), res.data.begin(), f);
						return resl
					}
			};
	}
	using tensor::Tensor;
}

using namespace TAT::legs_name;
using TAT::Tensor;
using TAT::Size;

int main() {
	auto t = Tensor<double>({2,3,4},{Up, Down, Left});
	t.generate([](){static int i=0; return i++;});
	for(Size i=0; i<2; i++){
		for(Size j=0; j<3; j++) {
			for(Size k=0; k<4; k++) {
				std::cout << t({{Up, i},{Down, j},{Left, k}}) << ' ';
			}
			std::cout << ',' << ' ';
		}
		std::cout << std::endl;
	}
	return 0;
}
